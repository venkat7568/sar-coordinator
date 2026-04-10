"""SAR Coordinator Environment — Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SARAction, SARObservation


class SAREnv(EnvClient[SARAction, SARObservation, State]):
    """
    Client for the SAR Coordinator Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance manages its own dedicated operational episode.

    Example:
        >>> with SAREnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset(task_id=1)
        ...     print(result.observation.mission_viability)
        ...
        ...     action = SARAction(action_type="deploy", resource_type="water")
        ...     result = env.step(action)
        ...     print(result.observation.sitrep)
    """

    def _step_payload(self, action: SARAction) -> Dict:
        """Serialize action — only send fields that are set (exclude None)."""
        return action.model_dump(exclude_none=True, exclude={"metadata"})

    def _parse_result(self, payload: Dict) -> StepResult[SARObservation]:
        """
        Parse server response into StepResult.

        Note: done and reward live at the top level of the payload,
        not inside the observation dict — the base class puts them there.
        """
        obs_data = payload.get("observation", {})
        observation = SARObservation(
            **obs_data,
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
