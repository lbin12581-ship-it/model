# 基于智能体的语音文本结构化总结系统

这是一个按功能文档实现的 Python Web 原型系统，面向会议转写总结与课堂知识提炼。

## 功能

- 文本粘贴与 TXT/DOCX 上传
- 文本清洗、去口语词、去重复
- 场景识别：会议、课堂、混合、通用
- 长文本分块处理
- 会议纪要、任务清单、风险问题、下一步计划
- 课堂知识点、重点难点、公式定理、解题思路、复习建议
- 质量校验与不确定信息提示
- Markdown、Word、PDF 导出
- 历史归档与关键词检索

## 运行

```bash
streamlit run app.py
```

浏览器打开 Streamlit 输出的本地地址即可使用。

## DeepSeek API 配置

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的DeepSeek_API_KEY
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
LLM_TEMPERATURE=0.2
LLM_TIMEOUT=90
```

系统会请求：

```text
https://api.deepseek.com/chat/completions
```

如果你想使用更强的模型，可以把模型改成：

```env
LLM_MODEL=deepseek-v4-pro
```

如果没有配置 `.env`，系统会继续使用本地规则智能体，不影响离线运行。

## 阿里云语音识别配置

在 `.env` 中加入：

```env
DASHSCOPE_API_KEY=你的阿里云百炼API_KEY
ASR_MODEL=paraformer-8k-v1
ASR_BASE_URL=https://dashscope.aliyuncs.com/api/v1
ASR_POLL_INTERVAL=3
ASR_TIMEOUT=600
ASR_DISFLUENCY_REMOVAL=true
ASR_DIARIZATION=false
```

页面中的“语音转文字（阿里云 Paraformer）”支持两种方式：

- 直接上传本地音频文件。系统会先通过 DashScope 临时文件上传接口得到 `oss://...` 临时 URL，再调用 `paraformer-8k-v1` 文件转写接口。
- 填写公网可访问的音频 URL。URL 需要能被阿里云服务器直接下载，不能有防盗链或登录限制。

## 说明

当前系统已经把大模型调用封装在 `summary_system/llm_client.py`，并在 `summary_system/agents.py` 中接入文本清洗、场景识别、会议总结、课堂知识提炼和质量校验智能体。阿里云语音识别封装在 `summary_system/asr_client.py`。API 调用失败时会自动回退到本地规则逻辑。
