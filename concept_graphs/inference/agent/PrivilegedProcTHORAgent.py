from typing import Dict, Any, Tuple, Union, List, Set, Optional

from collections import deque
import jax.numpy as jnp
import numpy as np

from langchain.agents import create_agent
from langchain.tools import tool
import logging

from concept_graphs.inference.agent.BaseProcTHORAgent import (
    BaseProcTHORAgent,
    AgentPrinterCallback,
)
from concept_graphs.utils import procthor_point_to_ros
from concept_graphs.inference.utils import NavigationResult
from semistaticsim.keyboardcontrol.main_skillsim import ROBOTS

log = logging.getLogger(__name__)


class PrivilegedProcTHORAgent(BaseProcTHORAgent):
    def __init__(self, model: str = "openai:gpt-5", **kwargs):
        super().__init__(**kwargs)
        self.llm_model = model
        self._llm_agent = None
        self.sim_time = self.sim.sss_data.self_at_current_time._timestamp.item()

    @property
    def llm_agent(self):
        if self._llm_agent is None:
            self._llm_agent = create_agent(model=self.llm_model, tools=self.tools())
        return self._llm_agent

    def tools(self):
        @tool
        def find_object(object_query: str) -> List[str]:
            """
            Finds objects matching the query.

            Returns a list of matching object IDs sorted by relevance.  If the returned
            list is empty, the object does not exist in the scene, and you should report
            it as not existent. Do not call this tool again for the same object.

            Args:
                object_query (str): Description of the object (e.g., "red apple").

            Returns:
                List[str]: Matching object IDs. Returns an empty list [] if no objects match the query.
            """
            return self.find_object(object_query)

        @tool
        def predict_object_receptacle(object_id: str) -> List[str]:
            """
            Computes the most probable receptacles (locations) for an object.

            If the first element of the list contains "ABSENT", the object is predicted to be absent from the scene,
            and you should report it as not existent. Do not call this tool again for the same object.
            The results of this tool should be trusted above any semantic prior you might have!

            Args:
                object_id (str): The unique ID of the object (e.g., "Laptop|Surface|2|3").

            Returns:
                List[str]: A list of the top_k most likely receptacle IDs for this object, sorted by probability.
                            If a specific candidate's probability is too low, it will be replaced 
                            by the string "ABSENT" to indicate it is not present.
                        
                            Example: ["Dresser|4|2", "ABSENT"] means the object is most likely at "Dresser|4|2", 
                            and the second best hypothesis is that the object is actually absent.
            """
            return self.predict_object_receptacle(object_id)

        @tool
        def navigate_to_receptacles(candidate_receptacles: List[str], looking_for_object: str) -> Dict[str, bool]:
            """
            Navigates to the candidate receptacles to search for a target object.

            The tool will attempt navigation to the `candidate_receptacles`. If the object is determined
            to be absent, navigation stops early and 'absent' is set to True.

            Args:
                candidate_receptacles (List[str]): The IDs of the target receptacles to approach (e.g., ["Dresser|1|2", "Bed|3|1"]), in order.
                looking_for_object (str): The ID of the object being searched for.

            Returns:
                Dict[str, Any]: A dictionary containing:
                    - "found": True if the object was found, False otherwise.
                    - "absent": True if the object was determined to be absent, False otherwise
            """
            result = self.navigate_to_receptacles(candidate_receptacles, looking_for_object)
            return {"found": result.found, "absent": result.is_absent}

        return [find_object, predict_object_receptacle, navigate_to_receptacles]

    def query(self, query: str):
        receptacle_names = ", ".join(self.toolbox.object_map.get_receptacles_name())
        system_message = {
            "role": "system",
            "content": f"""
You are an embodied LLM planner operating in a home environment.

**Environment Knowledge:**
- You know the static receptacles: <{receptacle_names}>.
- You do NOT know what objects are present or their unique IDs. 

**Standard Operating Procedure:**
1. **Ground the Object:** When a user mentions an object (e.g., "the red apple"), you MUST first use `find_object("red apple")` to retrieve its unique ID (e.g., "Apple|123") from all objects in the environment.
2. **Predict Locations:** Once you have the unique ID, call `predict_object_receptacle(id)` to get a list of likely receptacles sorted by probability.  
   - If the first returned element is "ABSENT", treat the object as absent.
3. **Navigate and Search (Only if requested):** If the user asks to "find", "check", or "go to" the object:
   - Call `navigate_to_receptacles(candidate_receptacles, looking_for_object)` with the predicted receptacles in order.
   - This tool automatically handles navigation attempts and stops when the object is found or determined absent.
   - Use the `found` and `absent` values from its output to decide your next response.

**Rules:**
- Never hallucinate object IDs. Always use `find_object` for grounding.
- Treat the first element "ABSENT" from any tool as **object absent**. Do not retry the same object unnecessarily.
- Trust the `predict_object_receptacle` tool over common sense.
- Only navigate when explicitly requested by the user.
- Always return structured, deterministic outputs. Use the `found` and `absent` signals from navigation tools to decide your next step.
""",
        }

        user_message = {"role": "user", "content": query}

        return self.llm_agent.invoke(
            {"messages": [system_message, user_message]},
            config={"callbacks": [AgentPrinterCallback(self.server)]},
        )

    #####################################
    # Utility methods used by the agent #
    #####################################
    def update(self, observation: Dict[str, Any]) -> None:
        """
        Updates the internal belief state based on a new environment observation.

        Args:
            observation (Dict[str, Any]): A dictionary containing the current state observations

        Returns:
            None
        """
        apn = observation["point_apn"]
        obj_centroid = procthor_point_to_ros(apn._aabb_center)
        assignments = apn._assignment
        object_containment_obs, object_centroids_obs = {}, {}
        for pickupable_idx, pickupable_id in enumerate(apn.pickupable_names):
            pickupable_vector = assignments[pickupable_idx]
            # Check if there is a positive observation
            found_positive = (pickupable_vector == 1.0).any()
            obs_centroid = {}
            if found_positive:
                obs_receptacle = {}
                # If a pickupable is seen in a receptacle, we can assume then it is not in other receptacles
                for idx, val in enumerate(pickupable_vector):
                    name = apn.receptacle_names[idx]
                    if name == "OOB_FAKE_RECEPTACLE":
                        continue
                    if val == 1.0:
                        obs_receptacle[name] = jnp.array([1.0])
                        obs_centroid[name] = obj_centroid[pickupable_idx]
                    elif val >= -1.0:
                        obs_receptacle[name] = jnp.array([0.0])
            else:
                # If not found, just return the places where it was not seen (if they exist)
                obs_receptacle = {
                    apn.receptacle_names[idx]: val
                    for idx, val in enumerate(pickupable_vector)
                    if val >= 0 and apn.receptacle_names[idx] != "OOB_FAKE_RECEPTACLE"
                }
            # Only populate if we have some observation
            if len(obs_receptacle) > 0:
                object_containment_obs[pickupable_id] = obs_receptacle
            if len(obs_centroid) > 0:
                object_centroids_obs[pickupable_id] = obs_centroid
        log.info(f"The object containment obs is: {object_containment_obs}")
        self.toolbox.temporal_map_update(object_containment_obs, self.current_time)
        self.toolbox.update_canonical_vectors(object_centroids_obs)

    def _oracle_is_object_visible(
        self, receptacle_id: str, looking_for_object: str, observation: Dict[str, Any]
    ) -> Union[bool, None]:
        """
        Checks if the current observation contains the pickupable of interest at the specified receptacle
        using privileged information.

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
        from semistaticsim.datawrangling.sssd import GeneratedSemiStaticData

        apn: GeneratedSemiStaticData = observation["point_apn"]

        vector_for_pickupable = apn._assignment[apn.pickupable_names.index(looking_for_object)]
        # Get the assignment for the target receptacle
        r_idx = apn.receptacle_names.index(receptacle_id)
        assigned_receptacle = vector_for_pickupable[r_idx]
        # Pickupable was observed!
        if (vector_for_pickupable == 1).any():
            # Is it in the right receptacle?
            if assigned_receptacle == 1:
                return True
            elif assigned_receptacle == 0:
                return False

        # Receptacle was observed but object not there
        if assigned_receptacle == 0:
            return False

        return None

    def found_pickupable(
        self, receptacle_id: str, looking_for_object: str, observation: Dict[str, Any]
    ) -> Union[bool, None]:
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
        from semistaticsim.datawrangling.sssd import GeneratedSemiStaticData

        apn: GeneratedSemiStaticData = observation["point_apn"]

        vector_for_pickupable = apn._assignment[apn.pickupable_names.index(looking_for_object)]
        # Get the assignment for the target receptacle
        r_idx = apn.receptacle_names.index(receptacle_id)
        assigned_receptacle = vector_for_pickupable[r_idx]
        # Pickupable was observed!
        if (vector_for_pickupable == 1).any():
            return True

        # If any of the receptacles in the list of receptacles for the object is observed, return False
        if (vector_for_pickupable == 0).any():
            return False

        return None

    ########################
    # Tool implementations #
    ########################
    def _find_object(self, object_query: str, top_k: int) -> List[str]:
        # Return all objects in the map and let the LLM pick the best match
        pickupables_name = self.toolbox.object_map.get_pickupables_name()
        return pickupables_name

    def _predict_object_receptacle(
        self,
        object_id: str,
        query_time: float,
        obj_threshold: float,
        top_k: int,
        add_noise: bool,
    ) -> List[str]:
        color = np.array([255, 0, 255])  # Magenta for absent visualization
        pickupable_name = self.resolve_query_into_pickupable(object_id)
        # Force to display object in last position it was seen, as it's an hypothesis add noise
        prediction, _ = self.toolbox.temporal_object_query(
            pickupable_name,
            query_time,
            threshold=obj_threshold,
            add_noise=add_noise,
            condition_on_object=True,
            strategy="last_seen",
        )

        # Sort candidates
        all_candidates = sorted(prediction, key=prediction.get, reverse=True)
        top_k_candidates = all_candidates[:top_k]

        # Return a list of receptacles above threshold or ABSENT
        final_list = []
        for cand in top_k_candidates:
            score = prediction[cand]
            if score >= obj_threshold:
                final_list.append(cand)
            else:
                final_list.append("ABSENT")

        display_target = "ABSENT"
        if final_list and final_list[0] != "ABSENT":
            display_target = final_list[0]
            
        if not self.stop_server:
            self.server.display_query_object(pickupable_name, display_target, color)

        return final_list

    def _decide_next_target(
        self,
        current_target: str,
        preds: List[str],
        visited: Set[str],
        must_switch: bool,
    ) -> Optional[str]:
        """
        Helper: Decides if we should switch targets based on predictions and visitation state.
        Returns the new target string, or None if we should stay/have no options.
        """
        # Target did not change
        if preds and preds[0] == current_target:
            return None

        best_unvisited = next((p for p in preds if p not in visited), None)

        if must_switch:
            return best_unvisited

        for p in preds:
            if p == best_unvisited:
                # Target changed!
                return best_unvisited
            if p == current_target:
                # Target did not change
                return None

        # Edge Case: The current target is no longer in the prediction list at all.
        if current_target not in preds:
            return best_unvisited

        return None

    def _visualize(self, obs, current_object, target):
        if not self.stop_server:
            self.server.display_agent(pose=obs["agent_pose"], img=obs["rgb_first_view"], intrinsics=obs["intrinsics"])
            self.server.display_query_object(current_object, target, np.array([255, 0, 255]))

    def _look_around(self, current_object, target):
        log.info("Looking around to find the object...")
        # Check if it's currently visible
        obs = self.render()
        self._reset_frame_cache()
        # Return present only if object is found and visible
        if self.found_pickupable(target, current_object, obs):
            return True
        # Reset cache to process new frame
        self._reset_frame_cache()
        cumulative_angle = 0
        # Scan Sequence: Left (-90) -> Front-Left (-45) -> Right (+90) -> Front-Right (+45)
        sequence = [-90, 45, 135, -45]
        for rot_angle in sequence:
            self.sim.RotateToAngle(ROBOTS[0], rot_angle, render_after=False)
            obs = self.render()
            cumulative_angle += rot_angle
            if self.found_pickupable(target, current_object, obs):
                # If object is found, no need to reset cache
                return True
            # Reset cache to process new frame
            self._reset_frame_cache()
        # Nothing found. Reset view to Center.
        self.sim.RotateToAngle(ROBOTS[0], -cumulative_angle, render_after=False)
        return False

    def _navigate_to_receptacles(
        self,
        candidate_receptacles: List[str],
        looking_for_object: str,
        do_update: bool = False,
        step_every: int = None,
        max_attempts: int = None,
        predict_every: int = None,
    ) -> Dict[str, Any]:
        """
        Unified navigation logic.

        If update_every is None: Visits candidate_receptacles in order.
        If update_every is not None:  Visits candidates, but periodically re-queries the model to
                        re-prioritize or add new targets to the queue.
        """
        from semistaticsim.rendering.floor import ReachabilityGraph

        # Initialize our target. This may change dynamically.
        target_queue = deque(candidate_receptacles)
        current_target = target_queue.popleft()
        # If determine as absent from first candidate, then stop navigation
        if current_target == "ABSENT":
            return NavigationResult(found=False, is_absent=True, nav_target=looking_for_object)

        visited_receptacles = {current_target}
        attempts = 0
        raw_iteration = 1
        remaining_path: List[Tuple[float, float, float]] = None
        nav_graph: ReachabilityGraph = None

        log.info(f"Starting search. Mode: {'Dynamic' if do_update else 'Static'}. First target: {current_target}")
        # Try N times to reach the receptacle
        while self.iterations < self.max_iterations:
            # This is procthor's reference frame!
            obj_size, obj_pos = self.toolbox.get_object_size_position(current_target, procthor=True)
            # Navigation step
            is_pathing_finished, remaining_path, _, geo_dist, nav_graph = self.sim.GoToNode(
                ROBOTS[0],
                obj_pos,
                obj_size,
                max_path_length=step_every,
                filtered_graph=nav_graph,
                precomputed_path_nodes=remaining_path,
            )
            found = None
            if is_pathing_finished:
                self.sim.RotateToNode(ROBOTS[0], obj_pos, tolerance=0.0)
                # There is no more waypoints, look around to find the object
                found = self._look_around(looking_for_object, current_target)

            # Render observation
            self.iterations += step_every
            obs = self.render()

            # --- CHECK SUCCESS ---
            found = self.found_pickupable(current_target, looking_for_object, obs) if found is None else found
            log.info(f"Was it found at {current_target}? {found}")
            # If receptacle is found to be visible but not the object, may be partial visibility
            # Avoid looking around twice if already done
            if found == False and not is_pathing_finished:
                self.sim.RotateToNode(ROBOTS[0], obj_pos, tolerance=0.0)
                found = self._look_around(looking_for_object, current_target)
                obs = self.render()

            # Visualize
            self._visualize(obs, looking_for_object, current_target)

            # Update internal state
            self.update(obs)
            self._reset_frame_cache()   # Clean cache after update
            # Check Ground truth state
            is_visible_oracle = self._oracle_is_object_visible(current_target, looking_for_object, obs)

            # If the object is found and visible, we are done
            if found is True and (geo_dist <= self.goal_thresh):
                if is_visible_oracle:
                    # Show latest state of scene
                    self.toolbox.refresh_object_map(condition_on_object=True, add_noise=False, strategy="last_seen")
                    self._visualize(obs, looking_for_object, current_target)
                    return NavigationResult(found=True, obs=obs, stats=self.sim.nav_metrics, nav_target=looking_for_object)
                else: 
                    found = False  # False alarm, object not visible or in wrong receptacle, try again

            # --- DECISION LOGIC ---
            # We must switch if: A) We arrived and didn't find it, or B) We confirmed it's not here.
            must_switch = is_pathing_finished or (found is False and geo_dist <= self.goal_thresh)
            next_target_candidate = None
            should_switch = False

            # === DYNAMIC MODE ===
            if do_update:
                # Predict every N iterations or if we must switch to avoid falling in static mode
                if (raw_iteration % predict_every == 0) or must_switch:
                    preds = self.predict_object_receptacle(looking_for_object)
                    # Only switch targets IF the target has been reached
                    preds = preds if (geo_dist <= self.goal_thresh) else [current_target]

                    # If a method returns an empty list, then it's a failure
                    if not preds:
                        return NavigationResult(found=False, obs=obs, stats=self.sim.nav_metrics, nav_target=looking_for_object)

                    # Use Helper to decide
                    next_target_candidate = self._decide_next_target(
                        current_target, preds, visited_receptacles, must_switch
                    )

                    # If helper returns a target, we switch.
                    # If helper returns None but must_switch is True, it means we ran out of targets.
                    if next_target_candidate:
                        should_switch = True
                    elif must_switch:
                        # Helper found nowhere to go (all predictions visited).
                        # This is a failure to find, not a confirmed absence.
                        return NavigationResult(
                            found=False,
                            obs=obs,
                            stats=self.sim.nav_metrics,
                            nav_target=looking_for_object,
                        )

            # === STATIC MODE ===
            else:
                if must_switch:
                    if target_queue:
                        next_target_candidate = target_queue.popleft()
                        should_switch = True
                    else:
                        return NavigationResult(found=False, obs=obs, stats=self.sim.nav_metrics, nav_target=looking_for_object)

            raw_iteration += 1

            # --- EXECUTE SWITCH ---
            if should_switch:
                # If next target is ABSENT, we are done
                if next_target_candidate == "ABSENT":
                    return NavigationResult(
                        found=False, is_absent=True, obs=obs, stats=self.sim.nav_metrics, nav_target=looking_for_object
                    )

                attempts += 1

                if max_attempts and attempts >= max_attempts:
                    return NavigationResult(
                        found=False, obs=obs, stats=self.sim.nav_metrics, nav_target=looking_for_object
                    )

                log.info(f"Switching: {current_target} -> {next_target_candidate}")

                current_target = next_target_candidate
                visited_receptacles.add(current_target)
                remaining_path = None
                nav_graph = None

        # Exit if max iterations exceeded
        return NavigationResult(
            found=False,
            obs=obs,
            stats=self.sim.nav_metrics,
            nav_target=looking_for_object,
        )

    def goto_object(self, pickupable_id: str) -> Tuple[Dict, bool]:
        """Closed-vocab nav to pickupables in the map"""
        # Reset nav metrics at the beginning of each query
        self.sim.reset_nav_metrics()
        # Fetch the most likely pickupable and get its receptacles
        list_of_receptacles = self.predict_object_receptacle(pickupable_id)
        if list_of_receptacles is None:
            return NavigationResult(found=False, is_absent=False, nav_target=pickupable_id)
        nav_results = self.navigate_to_receptacles(list_of_receptacles, pickupable_id)
        return nav_results
