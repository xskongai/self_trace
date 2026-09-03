Stage 1 — Data Collection
X 数据采集
↓
抓取自己的 Posts、文本、图片 URL、时间等

Stage 2 — Data Cleaning & Media Localization
数据清洗 + 媒体本地化
↓
去重、检查空 Post、下载图片、生成 enriched posts

Stage 3 — Per-Post Multimodal Analysis
单条 Post 多模态分析
↓
Text + Images
→ media_observations
→ post_understanding
→ evidence_signals

Stage 4 — Batch Analysis
全量批处理
↓
对 60 条 Post 批量执行 Stage 3
→ 每条一个结构化 JSON
→ usage / log / summary

Stage 5 — Temporal Aggregation
时间序列聚合
↓
把 60 条孤立 Post 串成“轨迹”
→ 主题归并
→ Signal 聚合
→ Frequency
→ Persistence
→ Recency
→ Emerging / Persistent / Fading

Stage 6 — Profile Generation
个人 Profile 生成
↓
从长期证据形成更高层结论
→ Interests
→ Knowledge
→ Activities
→ Thinking Patterns
→ Expression Style
→ Goals

Stage 7 — Evidence Trace & Presentation
证据追踪 + 最终展示
↓
每个 Profile Claim
→ 对应哪些 Posts
→ 哪个时间段
→ 什么 evidence
→ confidence
→ 为什么得到这个结论