# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Custom action managers used by the NoetixBumi tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from prettytable import PrettyTable

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionManager, ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg


class MyActionManager(ActionManager):
    """Manager for processing and applying actions for a given world.

    The action manager handles the interpretation and application of user-defined
    actions on a given world. It is comprised of different action terms that decide
    the dimension of the expected actions.

    The action manager performs operations at two stages:

    * processing of actions: It splits the input actions to each term and performs any
      pre-processing needed. This should be called once at every environment step.
    * apply actions: This operation typically sets the processed actions into the assets in the
      scene (such as robots). It should be called before every simulation step.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the action manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
        """
        if cfg is None:
            raise ValueError("Action manager configuration is None. Please provide a valid configuration.")

        manual_terms = getattr(env, "manual_action_terms", None)
        self._manual_terms: set[str] = set(manual_terms) if manual_terms is not None else set()
        self._manual_action_buffers: dict[str, torch.Tensor] = {}

        super().__init__(cfg, env)

        # keep only valid manual terms and warn if user requested missing entries
        missing_terms = [name for name in self._manual_terms if name not in self._terms]
        if missing_terms:
            print(
                f"[MyActionManager] Ignoring manual terms that are not present in the config: {missing_terms}"
            )
        self._manual_terms.intersection_update(self._terms.keys())

        self._manual_action_buffers = {
            name: torch.zeros((self.num_envs, self._terms[name].action_dim), device=self.device)
            for name in self._manual_terms
        }

        self._policy_term_names = [name for name in self._term_names if name not in self._manual_terms]

        # (re)create buffers to store policy actions only
        self._action = torch.zeros((self.num_envs, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

        # check if any term has debug visualization implemented
        self.cfg.debug_vis = False
        for term in self._terms.values():
            self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def total_action_dim(self) -> int:
        """Total dimension of policy-driven actions (excludes manual terms)."""
        if not hasattr(self, "_terms"):
            return 0
        return sum(
            term.action_dim for name, term in self._terms.items() if name not in self._manual_terms
        )

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each policy-controlled action term."""
        return [self._terms[name].action_dim for name in self._policy_term_names]

    @property
    def action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev_action

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    @property
    def get_IO_descriptors(self) -> list[dict[str, Any]]:
        """Get the IO descriptors for the action manager.

        Returns:
            A dictionary with keys as the term names and values as the IO descriptors.
        """

        data = []

        for term_name, term in self._terms.items():
            try:
                data.append(term.IO_descriptor.__dict__.copy())
            except Exception as e:
                print(f"Error getting IO descriptor for term '{term_name}': {e}")

        formatted_data = []
        for item in data:
            name = item.pop("name")
            formatted_item = {"name": name, "extras": item.pop("extras")}
            print(item["export"])
            if not item.pop("export"):
                continue
            for k, v in item.items():
                # Check if v is a tuple and convert to list
                if isinstance(v, tuple):
                    v = list(v)
                if k in ["description", "units"]:
                    formatted_item["extras"][k] = v
                else:
                    formatted_item[k] = v
            formatted_data.append(formatted_item)

        return formatted_data

    """
    Operations.
    """

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms: list[tuple[str, Sequence[float]]] = []
        idx = 0
        for name, term in self._terms.items():
            if name in self._manual_terms:
                term_actions = self._manual_action_buffers[name][env_idx].cpu()
            else:
                term_actions = self._action[env_idx, idx : idx + term.action_dim].cpu()
                idx += term.action_dim
            terms.append((name, term_actions.tolist()))
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the action data.
        Args:
            debug_vis: Whether to visualize the action data.
        Returns:
            Whether the debug visualization was successfully set. False if the action
            does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the action history
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        for buffer in self._manual_action_buffers.values():
            buffer[env_ids] = 0.0
        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        if self.total_action_dim != action.shape[1]:
            raise ValueError(
                f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}."
            )

        action = action.to(self.device)
        self._prev_action[:] = self._action
        self._action[:] = action

        idx = 0
        for name, term in self._terms.items():
            if name in self._manual_terms:
                term_actions = self._manual_action_buffers[name]
            else:
                term_actions = action[:, idx : idx + term.action_dim]
                idx += term.action_dim
            term.process_actions(term_actions)

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions()

    def get_term(self, name: str) -> ActionTerm:
        """Returns the action term with the specified name.

        Args:
            name: The name of the action term.

        Returns:
            The action term with the specified name.
        """
        return self._terms[name]

    def serialize(self) -> dict:
        """Serialize the action manager configuration.

        Returns:
            A dictionary of serialized action term configurations.
        """
        return {term_name: term.serialize() for term_name, term in self._terms.items()}

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._terms: dict[str, ActionTerm] = dict()

        cfg_items = self.cfg.items() if isinstance(self.cfg, dict) else self.cfg.__dict__.items()
        # parse action terms from the config
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term

    def set_manual_term_actions(self, term_name: str, values: torch.Tensor) -> None:
        """Update the cached actions for a manual term."""
        if term_name not in self._manual_terms:
            raise KeyError(f"Term '{term_name}' is not registered as a manual term.")
        buffer = self._manual_action_buffers[term_name]
        if values.shape != buffer.shape:
            raise ValueError(
                f"Manual term '{term_name}' expects values of shape {buffer.shape}, got {values.shape}."
            )
        buffer.copy_(values.to(self.device))
        self._terms[term_name].process_actions(buffer)
