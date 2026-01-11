import base64
import io
import json
import logging
import os
import sys

from PIL import Image

# Add the UI-TEMPR directory to sys.path to allow importing ui_tempr
UI_TEMPR_PATH = "/home/yuxiang/Documents/UI-TEMPR"
if UI_TEMPR_PATH not in sys.path:
    sys.path.append(UI_TEMPR_PATH)

try:
    from ui_tempr.tempr import Tempr
except ImportError:
    # Fallback or error if the path is wrong or dependencies are missing
    print(f"Error: Could not import ui_tempr from {UI_TEMPR_PATH}")
    Tempr = None

from mobile_world.agents.base import BaseAgent
from mobile_world.agents.utils.agent_mapping import QWENVL2AW_ACTION_MAP
from mobile_world.agents.utils.helpers import pil_to_base64
from mobile_world.runtime.utils.models import ENV_FAIL, JSONAction

logger = logging.getLogger(__name__)


class TemprAgent(BaseAgent):
    def __init__(
        self,
        model_name: str,
        llm_base_url: str,
        api_key: str = "empty",
        max_try: int = 5,
        *args,
        os_environ: dict = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if Tempr is None:
            raise ImportError(
                "ui_tempr module not found. Please ensure the path is correct."
            )

        self.tempr = Tempr(
            model=model_name,
            base_url=llm_base_url,
            api_key=api_key,
            max_try=max_try,
        )
        # TEMPR state management
        self.subtasks = None
        self.essential_states = None
        self.current_subtask = None
        self.current_state = None

        # History tracking for TEMPR requirements
        self.history_screenshots = []
        self.history_actions = []
        self.history_agent_messages = []

        self.instruction = None

    def initialize(self, instruction: str) -> bool:
        super().initialize(instruction)
        # Reset TEMPR state on initialization
        self.subtasks = None
        self.essential_states = None
        self.current_subtask = None
        self.current_state = None
        self.history_screenshots = []
        self.history_actions = []
        self.history_agent_messages = []
        self.tempr.reset()
        return True

    def predict(self, observation: dict) -> tuple[str, JSONAction]:
        if self.instruction is None:
            raise ValueError("Agent not initialized")

        screenshot_pil = observation["screenshot"]
        current_screenshot_b64 = pil_to_base64(screenshot_pil)

        # New simplified logic using Tempr.predict
        action_dict_raw, action_instruction = self.tempr.predict(
            self.instruction, current_screenshot_b64
        )

        if action_instruction is None:
            return "Failed to generate plan", JSONAction(
                action_type=ENV_FAIL, text="Failed to generate plan"
            )

        if action_dict_raw is None:
            # Could be termination or failure
            if action_instruction and "finished" in action_instruction.lower():
                # Handle success/fail termination signal from predict
                return action_instruction, JSONAction(
                    action_type=QWENVL2AW_ACTION_MAP["terminate"],
                    text=action_instruction,
                )
            return "Failed to execute plan", JSONAction(
                action_type=ENV_FAIL, text="Failed to execute plan"
            )

        try:
            # handle action_dict parsing
            if "arguments" in action_dict_raw:
                action_args = action_dict_raw["arguments"]
            else:
                action_args = action_dict_raw
        except Exception as e:
            logger.error(f"Error parsing action arguments: {e}")
            return f"Error parsing action: {e}", JSONAction(
                action_type=ENV_FAIL, text="Error parsing action arguments"
            )

        # 4. Convert to MobileWorld Action
        width, height = screenshot_pil.size
        # The mw_action_dict conversion relies on action_args
        mw_action_dict = self._to_mobile_world_action(action_args, width, height)

        # Store history (MW legacy)
        self.history_actions.append(action_args)
        self.history_agent_messages.append(action_instruction)
        self.history_screenshots.append(current_screenshot_b64)

        return action_instruction, JSONAction(**mw_action_dict)

    def _to_mobile_world_action(
        self, action_dict: dict, width: int, height: int
    ) -> dict:
        """
        Transform the TEMPR action dictionary to MobileWorld JSONAction format.
        Based on AITK's to_device method.
        """
        action_type = action_dict.get("action")

        if action_type == "click":
            x, y = action_dict.get("coordinate", [None, None])
            if x is None or y is None:
                return {"action_type": "unknown", "text": "Click not parsed"}
            # Scale coordinates (TEMPR uses 1000x1000 normalization)
            x_abs = int(x / 1000 * width)
            y_abs = int(y / 1000 * height)
            return {
                "action_type": QWENVL2AW_ACTION_MAP["click"],
                "x": x_abs,
                "y": y_abs,
            }

        elif action_type == "long_press":
            x, y = action_dict.get("coordinate", [None, None])
            if x is None or y is None:
                return {"action_type": "unknown", "text": "Long Press not parsed"}
            x_abs = int(x / 1000 * width)
            y_abs = int(y / 1000 * height)
            return {
                "action_type": QWENVL2AW_ACTION_MAP["long_press"],
                "x": x_abs,
                "y": y_abs,
            }

        elif action_type == "swipe":
            if action_dict.get("coordinate") and action_dict.get("coordinate2"):
                x1, y1 = action_dict.get("coordinate")
                x2, y2 = action_dict.get("coordinate2")
                # Scale
                start_x = int(x1 / 1000 * width)
                start_y = int(y1 / 1000 * height)
                end_x = int(x2 / 1000 * width)
                end_y = int(y2 / 1000 * height)

                return {
                    "action_type": QWENVL2AW_ACTION_MAP["swipe"],
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                }
            elif action_dict.get("direction"):
                # Handle directional swipe map to coordinates
                direction = action_dict.get("direction")
                if direction == "up":
                    start_x, start_y = width // 2, height * 3 // 4
                    end_x, end_y = width // 2, height // 4
                elif direction == "down":
                    start_x, start_y = width // 2, height // 4
                    end_x, end_y = width // 2, height * 3 // 4
                elif direction == "left":
                    start_x, start_y = width * 4 // 5, height // 2
                    end_x, end_y = width // 5, height // 2
                elif direction == "right":
                    start_x, start_y = width // 5, height // 2
                    end_x, end_y = width * 4 // 5, height // 2
                else:
                    return {
                        "action_type": "unknown",
                        "text": f"Swipe direction {direction} not supported",
                    }

                return {
                    "action_type": QWENVL2AW_ACTION_MAP["swipe"],
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                }
            else:
                return {"action_type": "unknown", "text": "Swipe parameters missing"}

        elif action_type == "type":
            text = action_dict.get("text", "")
            return {"action_type": QWENVL2AW_ACTION_MAP["type"], "text": text}

        elif action_type == "system_button":
            button = action_dict.get("button", "")
            if button == "Back":
                return {"action_type": QWENVL2AW_ACTION_MAP["back"]}
            elif button == "Home":
                return {"action_type": QWENVL2AW_ACTION_MAP["home"]}
            elif button == "Enter":
                return {"action_type": QWENVL2AW_ACTION_MAP["enter"]}
            else:
                return {
                    "action_type": "unknown",
                    "text": f"System button {button} not supported",
                }

        elif action_type == "open":
            app_name = action_dict.get("text", "")
            return {"action_type": "open_app", "app_name": app_name}

        elif action_type == "wait":
            return {"action_type": QWENVL2AW_ACTION_MAP["wait"]}

        elif action_type == "terminate":
            status = action_dict.get("status", "")
            return {"action_type": QWENVL2AW_ACTION_MAP["terminate"], "text": status}

        elif action_type == "answer":
            return {
                "action_type": QWENVL2AW_ACTION_MAP["answer"],
                "text": action_dict.get("answer", ""),
            }

        elif action_type == "error":
            return {
                "action_type": "unknown",
                "text": action_dict.get("answer", "Error action"),
            }

        else:
            return {
                "action_type": "unknown",
                "text": f"Unknown action type: {action_type}",
            }
