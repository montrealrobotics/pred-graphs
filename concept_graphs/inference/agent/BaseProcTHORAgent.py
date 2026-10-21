import time
from abc import ABC
from typing import Dict, Any, List

from semistaticsim.rendering.simulation.skill_simulator import Simulator

from concept_graphs.viz.server.ProcTHORServer import PrivilegedServer
from concept_graphs.inference.toolbox.PerpetuaMapToolbox import PerpetuaMapToolbox

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

import logging

log = logging.getLogger(__name__)


class BaseProcTHORAgent(ABC):
    def __init__(
        self,
        sim: Simulator,
        server: PrivilegedServer,
        top_k: int = 3,
        max_attempts: int = 3,
        goal_thresh: float = 1.0,
        step_every: int = 5,
        do_update: bool = False,
        predict_every: int = 1,
        obj_threshold: float = 0.5,
        max_iterations: int = 1000,
        **kwargs,
    ):
        # Class objects
        self.sim = sim
        self.server = server

        # Attributes
        self.max_iterations = max_iterations  # Maximum number of iterations allowed
        self.dt = 1 / max_iterations
        self.iterations = 0  # To track the number of updates
        self.sim_time = None  # To track the simulation time at start

        # Hyperparameters
        self.top_k = top_k
        self.max_attempts = max_attempts
        self.goal_thresh = goal_thresh
        self.step_every = step_every
        self.do_update = do_update
        self.predict_every = predict_every
        self.obj_threshold = obj_threshold

    @property
    def stop_server(self):
        return self.server.stop_server

    @property
    def current_time(self):
        return self.sim_time + self.dt * self.iterations

    @property
    def toolbox(self) -> PerpetuaMapToolbox:
        return self.server.toolbox

    def _reset_frame_cache(self):
        pass

    def resolve_query_into_pickupable(self, query: str) -> str:
        # Assumes query is a cleaned up name
        for p in self.toolbox.object_map.get_pickupables_name():
            if p.startswith(query):
                return p

    def render(self) -> Dict[str, Any]:
        """
        Renders the current observation from the simulator.

        Returns:
            Dict[str, Any]: A dictionary containing the current state observations.
        """
        obs = self.sim.render()
        self.sim.privileged_apn = None
        return obs

    def update(self, observation: Dict[str, Any]) -> None:
        """
        Updates the internal belief state based on a new environment observation.

        Args:
            observation (Dict[str, Any]): A dictionary containing the current state observations

        Returns:
            None
        """
        raise NotImplementedError()

    def found_pickupable(self, receptacle_id: str, looking_for_object: str, observation: Dict[str, Any]) -> bool | None:
        """
        Checks if the current observation contains the pickupable of interest at the specified receptacle.

        Args:
            receptacle_id (str): The id of the target receptacle.
            looking_for_object (str): The id of the object we are searching for.
            observation (Dict[str, Any]): The observation dictionary for the current state.

         Returns:
            bool | None:
                - True if the object is detected at the specified receptacle.
                - False if the receptacle is visible but the object is not present at that receptacle.
                - None if the receptacle is not visible in the current observation.
        """
        raise NotImplementedError()

    def _find_object(self, object_query: str, top_k: int) -> List[str]:
        raise NotImplementedError()

    def find_object(self, object_query: str, top_k: int = None) -> List[str]:
        """
        Finds objects matching the query.

        Returns a list of matching object IDs sorted by relevance.  If the returned
        list is empty, the object does not exist in the scene, and you should report
        it as not existent.

        Args:
            object_query (str): Description of the object (e.g., "red apple").
            top_k (int): The maximum number of results to return.

        Returns:
            List[str]: Matching object IDs. Returns an empty list [] if no objects match the query.
        """
        if top_k is None:
            top_k = self.top_k
        return self._find_object(object_query, top_k)

    def _predict_object_receptacle(
        self,
        object_id: str,
        query_time: float,
        obj_threshold: float,
        top_k: int,
        add_noise: bool,
    ) -> List[str]:
        raise NotImplementedError()

    def predict_object_receptacle(
        self,
        object_id: str,
        query_time: float = None,
        obj_threshold: float = None,
        top_k: int = None,
        add_noise: bool = True,
    ) -> List[str]:
        """
        Computes the most probable receptacles (locations) for an object at a specific time.

        If the returned list returns "ABSENT", the object is predicted to be absent from the scene,
        and you should report it as not existent. Do not call this tool again for the same object.

        Args:
            object_id (str): The unique ID of the object (e.g., "Laptop|Surface|2|3").
            query_time (float): The time for which the prediction should be made.
            obj_threshold (float): The threshold to determine whether the object is present.
            top_k (int): The maximum number of receptacles to return.

        Returns:
            List[str]: A list of up to `top_k` most likely receptacle IDs.
                       If a candidate's probability is below `obj_threshold`, it is replaced
                       by the string "ABSENT" in the list.

                       Example: ["Table", "ABSENT"] means the object is most likely at "Table",
                       and the second best hypothesis was rejected by the threshold.
        """

        if query_time is None:
            query_time = self.current_time
        if top_k is None:
            top_k = self.top_k
        if obj_threshold is None:
            obj_threshold = self.obj_threshold

        return self._predict_object_receptacle(object_id, query_time, obj_threshold, top_k, add_noise)

    def _navigate_to_receptacles(
        self,
        candidate_receptacles: List[str],
        looking_for_object: str,
        do_update: bool,
        step_every: int,
        max_attempts: int,
        predict_every: int,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def navigate_to_receptacles(
        self,
        candidate_receptacles: List[str],
        looking_for_object: str,
        do_update: bool = None,
        step_every: int = None,
        max_attempts: int = None,
        predict_every: int = None,
    ) -> Dict[str, Any]:
        """
        Navigates to the candidate receptacles to search for a target object.

        The tool will attempt navigation up to `max_attempts` times in the order of `candidate_receptacles`.
        If the object is determined to be absent, navigation stops early and 'absent' is set to True.

        Args:
            candidate_receptacles (List[str]): The IDs of the target receptacles to approach
                (e.g., ["Dresser|1|2", "Bed|3|1"]), in order.
            looking_for_object (str): The ID of the object being searched for (e.g., "Laptop|Surface|2|3").
            do_update (bool): Whether to update the internal belief state after each observation during navigation.
            step_every (int): How many simulation steps to take before rendering a new observation.
            max_attempts (int): The maximum number of navigation attempts.
            predict every (int): If in update mode, how often to re-predict the object's location.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "obs": The last observation dictionary obtained.
                - "found": True if the object was found, False otherwise.
                - "absent": True if the object was determined to be absent, False otherwise
        """
        if do_update is None:
            do_update = self.do_update
        if predict_every is None:
            predict_every = self.predict_every
        if step_every is None:
            step_every = self.step_every
        if max_attempts is None:
            max_attempts = self.max_attempts

        return self._navigate_to_receptacles(
            candidate_receptacles, looking_for_object, do_update, step_every, max_attempts, predict_every
        )

    def spin(self):
        while True:
            self._callbacks()
            time.sleep(0.05)

    def _callbacks(self):
        # LLM
        if self.server.open_vocab_query is not None:
            self.query(self.server.open_vocab_query)
            self.server.open_vocab_query = None
        # Predict
        if self.server.object_query_time is not None and self.server.selected_object_id is not None:
            self.predict_object_receptacle(self.server.selected_object_id, self.server.object_query_time)
            self.server.object_query_time = None
            self.server.selected_object_id = None
            self.server.receptacle_id = None
        # Go to receptacle
        if self.server.selected_object_id is not None and self.server.receptacle_id is not None:
            self.navigate_to_receptacles([self.server.receptacle_id], self.server.selected_object_id)
            self.server.selected_object_id = None
            self.server.receptacle_id = None
        if self.server.clip_query != "":
            self.find_object(self.server.clip_query)
            self.server.clip_query = ""


class AgentPrinterCallback(BaseCallbackHandler):
    """
    Custom callback to print LLM reasoning and tool usage to the console.
    """

    def __init__(self, server=None):
        super().__init__()
        self.server = server

    def on_chat_model_start(self, serialized, messages, **kwargs):
        pass

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Print the LLM's reasoning or text response."""
        # Access the actual text generated
        if response.generations and response.generations[0]:
            generation = response.generations[0][0]
            # Check if it's a ChatGeneration (standard for OpenAI)
            if hasattr(generation, "message"):
                content = generation.message.content
                # Only print if there is actual text (sometimes it's just a tool call)
                if content:
                    if not self.server.stop_server:
                        self.server.display_llm_response(f"[LLM Reasoning]: {content}\n")
                    log.info(f"\n\033[1m[LLM Reasoning]\033[0m:\n{content}\n")
            # Fallback for standard generation
            elif generation.text:
                if not self.server.stop_server:
                    self.server.display_llm_response(f"[LLM Reasoning]: {generation.text}\n")
                log.info(f"\n\033[1m[LLM Reasoning]\033[0m:\n{generation.text}\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Print which tool is being called and with what inputs."""
        tool_name = serialized.get("name")
        if not self.server.stop_server:
            self.server.display_llm_response(f"[Tool Call]: {tool_name}\n[Arguments]: {input_str}")
        log.info(f"\033[94m[Tool Call]\033[0m: {tool_name}")
        log.info(f"\033[94m[Arguments]\033[0m: {input_str}")

    def on_tool_end(self, output, **kwargs):
        """Print the result from the tool."""
        if not self.server.stop_server:
            self.server.display_llm_response(f"[Tool Output]: {str(output)}\n")
        log.info(f"\033[92m[Tool Output]\033[0m: {str(output)}\n")
