"""Ask tool-only models for a public action summary in the same model call."""
from copy import deepcopy

SUMMARY_ARG = "_action_summary"


def build_action_progress_middleware():
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.utils.function_calling import convert_to_openai_tool

    class ActionProgressMiddleware(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            tools = []
            for tool in request.tools:
                schema = deepcopy(convert_to_openai_tool(tool))
                function = schema.get("function")
                if function is not None:
                    parameters = function.setdefault("parameters", {"type": "object"})
                    properties = parameters.setdefault("properties", {})
                    properties[SUMMARY_ARG] = {
                        "type": "string",
                        "description": (
                            "一句简短、面向用户的行动说明：结合当前问题和已有工具结果，"
                            "说明这次具体要查什么或核对什么；不输出内部推理，不虚构执行结果。"
                        ),
                    }
                    required = parameters.setdefault("required", [])
                    if SUMMARY_ARG not in required:
                        required.append(SUMMARY_ARG)
                tools.append(schema)
            return handler(request.override(tools=tools))

        def wrap_tool_call(self, request, handler):
            # Presentation metadata must not reach tool validation, caches or business logic.
            args = dict(request.tool_call.get("args") or {})
            args.pop(SUMMARY_ARG, None)
            return handler(request.override(tool_call={**request.tool_call, "args": args}))

    return ActionProgressMiddleware()
