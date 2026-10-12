import asyncio, json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama

import os
from gguf import GGUFReader
from jinja2 import Environment

import ollama

async def main():
    #========실습 1단계==========
    print("========실습 1단계==========")
    client = MultiServerMCPClient({"colab": {
        "transport": "stdio", "command": "colab-mcp", "args": []}})
    tools = await client.get_tools()

    # 1) 모델에게 주입될 도구 정의 JSON 출력
    print(json.dumps([convert_to_openai_tool(t) for t in tools],
                     ensure_ascii=False, indent=1))

    # 2) 정의된 도구를 연결해 실제 요청
    llm = ChatOllama(model="qwen3.5:2b", temperature=0, reasoning=False)
    r = await llm.bind_tools(tools).ainvoke("구글 코랩을 브라우저로 연결해 줘")
    print("tool_calls:", [(tc["name"], tc["args"]) for tc in r.tool_calls])
    print("prompt_eval_count:", r.response_metadata.get("prompt_eval_count"))

    #========실습 2단계==========
    print("========실습 2단계==========")
    man = json.load(open(os.path.expanduser(
        "~/.ollama/models/manifests/registry.ollama.ai/library/qwen3.5/2b")))
    digest = next(l["digest"] for l in man["layers"] if "model" in l["mediaType"])
    blob = os.path.expanduser("~/.ollama/models/blobs/" + digest.replace(":", "-"))
    field = GGUFReader(blob).get_field("tokenizer.chat_template")
    template = bytes(max(field.parts[1:], key=len)).decode("utf-8")

    env = Environment()
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    rendered = env.from_string(template).render(
        messages=[{"role": "user", "content": "구글 코랩을 브라우저로 연결해 줘"}],
        tools=[convert_to_openai_tool(t) for t in tools],
        add_generation_prompt=True)
    print(rendered)

    #========실습 3단계==========
    print("========실습 3단계==========")
    g = ollama.generate(model="qwen3.5:2b", prompt=rendered, raw=True,
                        options={"num_predict": 1})
    print("raw prompt_eval_count:", g.get("prompt_eval_count"))

asyncio.run(main())