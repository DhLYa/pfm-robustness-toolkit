# This file has been modified from the original reetoolbox project.
# Additional changes by Dhyey Yajnik, 2025.

from abc import ABC, abstractmethod
import torch
from reetoolbox.constraints import Constraints
import time


def untargeted_loss(outputs, labels):
    loss = outputs.gather(1, labels.unsqueeze(1))[:, 0]
    return loss


def targeted_loss(outputs, targets):
    num_classes = list(outputs.shape)[1]
    all_out = torch.sum(outputs, dim=1)
    target_out = outputs.gather(1, targets)[:, 0]
    loss = (all_out / num_classes) - target_out
    # loss = -target_out
    return loss


class Optimiser(ABC):
    def __init__(self, model, Transform, hyperparameters, transform_hyperparameters, criterion=untargeted_loss,
                 device="cuda:0"):
        self.model = model
        self.Transform = Transform
        self.hyperparameters = hyperparameters
        self.transform_hyperparameters = transform_hyperparameters
        self.device = device
        self.transform = None
        self.criterion = criterion

    @abstractmethod
    def optimise(self):
        pass

import time
import torch

class PGD(Optimiser):
    def optimise(self, inputs, targets=None, reset_weights=True):
        start_time = time.time()
        
        epsilon   = self.hyperparameters["epsilon"]
        steps     = self.hyperparameters["steps"]
        constraint= self.hyperparameters["constraint"]
        C         = self.hyperparameters["C"]
        input_range = self.hyperparameters["input_range"]

        if constraint is not None:
            constraint_func = getattr(Constraints, constraint)
            constraints = Constraints

        # Move inputs to device
        inputs = inputs.to(self.device)

        # (Re)initialize transform
        if self.transform is None or reset_weights:
            t_init = time.time()
            self.transform = self.Transform(
                input_shape=inputs.shape,
                device=self.device,
                **self.transform_hyperparameters
            )
            # print(f"Transform initialization took: {time.time() - t_init:.2f} s")

        # print(f"Initial setup time: {time.time() - start_time:.2f} s")

        # Switch model to eval, save original mode
        in_train_mode = self.model.training
        # print("Before eval():", self.model.training)
        self.model.eval()
        # print("After eval():", self.model.training)

        # Freeze model params
        grads = []
        for p in self.model.parameters():
            grads.append(p.requires_grad)
            p.requires_grad = False

        # Optimiser for transform weights
        opt = torch.optim.RMSprop([self.transform.weights], lr=epsilon)

        # Compute targets once
        if targets is None:
            t_tgt = time.time()
            with torch.no_grad():
                out = self.model(inputs)
                targets = torch.argmax(out, dim=1)
            # print(f"Target computation time: {time.time() - t_tgt:.2f} s")

        total_step_time = 0.0

        # Main PGD loop
        for i in range(steps):
            step_start = time.time()
            opt.zero_grad()

            # Transform forward
            t_fwd = time.time()
            adv_inputs = self.transform.forward(inputs)
            t_fwd_time = time.time() - t_fwd

            # Model forward
            t_mf = time.time()
            adv_outputs = self.model(adv_inputs)
            t_mf_time = time.time() - t_mf

            # Loss and reduction
            loss = self.criterion(adv_outputs, targets)
            # print(f"  Raw loss shape: {tuple(loss.shape)}")
            if loss.dim() > 0:
                loss = loss.mean()
            # print(f"  Reduced loss value: {loss.item():.4f}")

            # Backward
            t_bw = time.time()
            loss.backward()
            t_bw_time = time.time() - t_bw

            # Step
            t_step = time.time()
            opt.step()
            t_step_time = time.time() - t_step

            # Constraint
            if constraint is not None:
                t_con = time.time()
                self.transform.weights = constraint_func(
                    constraints,
                    self.transform.weights,
                    self.transform.base_weights,
                    C
                )
                t_con_time = time.time() - t_con
            else:
                t_con_time = 0.0

            step_time = time.time() - step_start
            total_step_time += step_time

            # print(
            #     f"Step {i+1}/{steps}: "
            #     f"Transform: {t_fwd_time:.2f}s, "
            #     f"Model: {t_mf_time:.2f}s, "
            #     f"Backward: {t_bw_time:.2f}s, "
            #     f"Opt.step: {t_step_time:.2f}s, "
            #     f"Constraint: {t_con_time:.2f}s, "
            #     f"Total: {step_time:.2f}s"
            # )

        # print(f"Total optimisation loop time: {total_step_time:.2f} s")

        # Final forward
        t_fin = time.time()
        original_inputs = inputs.clone()
        adv_inputs = self.transform.forward(inputs)
        # print(f"Final forward pass time: {time.time() - t_fin:.2f} s")

        # Restore model params
        for idx, p in enumerate(self.model.parameters()):
            p.requires_grad = grads[idx]
        if in_train_mode:
            self.model.train()

        # print(f"PGD optimisation completed in: {time.time() - start_time:.2f} s")
        return inputs, adv_inputs


class StochasticSearch(Optimiser):
    def optimise(self, inputs, targets=None, reset_weights=True):
        samples = self.hyperparameters["samples"]
        weight_ranges = self.hyperparameters["weight_ranges"]
        input_range = self.hyperparameters["input_range"]

        inputs = inputs.to(self.device)

        if self.transform is None or reset_weights:
            self.transform = self.Transform(input_shape=inputs.shape, device=self.device,
                                            **self.transform_hyperparameters)
            self.best_loss = None
            self.best_adv_inputs = inputs

        in_train_mode = self.model.training
        self.model.eval()

        grads = []
        for param in self.model.parameters():
            grads.append(param.requires_grad)
            param.requires_grad = False

        with torch.no_grad():
            for i in range(samples):
                for j, weight_name in enumerate(weight_ranges):
                    self.transform.weights[weight_name] = torch.FloatTensor(
                        *self.transform.weights[weight_name].shape).uniform_(*weight_ranges[weight_name]).to(self.device)

                if targets is None:
                    outputs = self.model(inputs)
                    targets = torch.argmax(outputs, dim=1)

                original_inputs = inputs.clone()
                adv_inputs = self.transform.forward(inputs)
                inputs = original_inputs
                adv_outputs = self.model(adv_inputs)

                loss = self.criterion(adv_outputs, targets)

                if self.best_loss is None or self.best_adv_inputs is None:
                    self.best_loss = loss.clone()
                    self.best_adv_inputs = adv_inputs.clone()
                else:
                    for j, input_loss in enumerate(loss):
                        if input_loss < self.best_loss[j]:
                            self.best_adv_inputs[j] = adv_inputs[j]
                            self.best_loss[j] = input_loss

        original_inputs = inputs.clone()
        inputs = original_inputs

        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        if in_train_mode:
            self.model.train()

        return inputs, self.best_adv_inputs
