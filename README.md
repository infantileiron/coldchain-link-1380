# 鲜达冷链 · 温控与损耗关联簿

给鲜达冷链 12 条线路的运营分析组，用运输背景解释温控与货损之间的关联。上游班次和温控记录进入工作簿，分析员检查缺失、执行真实统计，负责人把改善建议签核给线路主管。

原流程是调度员整理工作簿，分析员复制公式，主管在聊天记录里拍板。三个痛点是缺失样本被静默删除、统计口径容易改错、建议与研究证据分离。此项目提供班次清单、样本质检、关联分析、改善建议、货损预警五个业务页面；并含登录、审计、CSV 之外的 JSON 研究导出。

关联分析回答“温控与货损是否相关”；货损预警用同一批温控特征训练货损率预测，给尚未填写货损的新样本一个可解释的风险预估值。

## 启动

```sh
cp .env.example .env
docker compose up -d --build
docker compose exec -T backend python -m scripts.seed_demo
```

访问 http://localhost:18201 ，后端 http://localhost:19201/docs ，MySQL 映射 17701。恰好五个健康检查服务 frontend/backend/worker/db/redis。backend 与 worker 共享 `/artifacts`；Redis appendonly；数据库命名卷。停止与清空演示数据：`docker compose down -v --remove-orphans`。

## 演示账号和权限

| 邮箱 | 密码 | 权限 |
|---|---|---|
| admin@demo.local | Demo123! | 全部读写、负责人签核与审计 |
| analyst@demo.local | Demo123! | 创建和修改班次/样本、提交研究、起草建议；不能签核 |
| viewer@demo.local | Demo123! | 所有业务数据只读与下载；禁止任何写入和计算任务提交 |

JWT 有效期 8 小时，后端查数据库角色，不信任前端传入角色。密码 PBKDF2；演示使用固定盐和可公开的 JWT key，部署时通过环境变量替换 key，并为正式账号改成每用户随机盐。首次启动 SEED_DEMO=true 自动导入；失败记日志。自然键唯一约束、只插入缺失数据，重复执行不覆盖用户修改。

## 一天的业务剧本

1. 08:50 viewer 核对昨日 XD-1115-001 班次。
2. 09:00 analyst 查看 60 班次、60 条样本和缺失数据。
3. 09:10 创建 XD-1115-061，鲜达线路 01 / 冷藏。
4. 09:20 采集暴露 32 分钟、超温 3 次、货损 4.2% 的样本。
5. 09:30 创建冷藏温区暴露时长研究。
6. 09:40 Redis 队列交给 worker 计算；刷新查看真实结果。
7. 10:00 阅读散点、p 值、配对数及混杂因素警示。
8. 10:10 起草“复核线路 01 装卸等待时长”等建议，admin 签核。
9. 10:20 viewer 下载 coldchain-link-<研究id>.json 供线路主管复核。

## 算法与边界

Python SciPy 计算 Pearson r = sum((x-xmean)(y-ymean))/sqrt(sum((x-xmean)^2)sum((y-ymean)^2))；Spearman 使用平均秩处理并列值，再对秩求 Pearson。返回双侧 p 值、有效配对数、排除数、散点和原始快照。输入为同温区样本，x 为暴露分钟或超温次数，y 固定为货损率%。只有 x/y 均非空的配对进入计算；不做插补。少于3对、任一变量为常量，任务明确 failed。相关性不等于因果，不自动决定线路处罚。60条 seed 使用 Random(3102)；冷藏和冷冻各有一条缺失暴露，10条常温对照的暴露均20分钟，用于失败诊断。研究创建时冻结输入，后续修改样本不改变已有结果。任务失败可修复原始记录再新建研究。worker 启动恢复 queued/running 任务；单 worker 模式，不承诺多副本 exactly-once。

## 货损预警模型

第五个页面“货损预警”负责训练、对比、上线、回滚；样本质检页可对未填货损的样本直接生成预警。

- 训练输入在训练时冻结进版本快照（含每行样本、班次、切分归属和输入指纹）；之后修改或新增样本不会改写已有版本。
- 按班次分组切分训练集/留出集（`random.Random(seed)` 打乱班次，默认留出 20%）。同一个班次的所有样本只属于一个集合，页面展示两边的班次清单。
- 缺失处理与标准化只从训练集学习：暴露时长/超温次数的均值既用于填补缺失特征，也用于标准化；留出集只应用这些参数，不参与学习。货损缺失的样本保留在冻结快照里但不进入拟合与留出评估。
- 模型为两特征线性回归（暴露时长、超温次数）。基线是训练集货损率均值，在同一批留出样本上比较 MAE。留出 MAE 未严格优于基线的版本标记 `rejected`，接口拒绝上线。
- 预测使用当前唯一上线版本，单笔预测保存版本号、输入指纹和参数副本；之后上线或回滚都不改写历史预测，单笔预测不会混用版本。未填货损的样本同样可预测，缺失特征按该版本冻结的填补值处理，风险原因会注明“按训练集均值填补”。
- 发布流程：写版本模型文件 → 写上线记录与指针 → 数据库提交；任一写入失败即恢复旧指针并回滚数据库，旧版本保持上线。`request_key`（或 `Idempotency-Key` 头）唯一约束保证失败后重试不重复发布。并发上线用数据库租约锁 + 当前版本 CAS，只接受一个版本。
- 回滚只能回到曾经上线过的版本，只移动指针，不复制/改写模型文件；回滚后新样本使用恢复版本，历史预测与历史研究仍可读取。
- 模型文件与指针位于 `/artifacts/models`（backend 与 worker 共享卷）；数据库指针是事实来源，后端启动时会校正指针文件。

接口：`GET /models`（对比与上线记录）、`POST /models/train`、`GET /models/{id}`、`POST /models/{id}/deploy`、`POST /models/rollback`、`GET /models/deployments`、`POST /predictions`、`GET /predictions`、`GET /predictions/{id}`。训练与上线需要 analyst/admin，viewer 只读。

## 技术和目录

Python3.11 / FastAPI / SQLAlchemy / SciPy / PyMySQL / PyJWT；MySQL8.0；Redis7；Nginx1.27；原生 HTML、CSS、JS ES modules。前端无构建步骤。

- backend/src/models、schemas、services、controllers、routes 分层承载实体与业务。
- backend/src/middlewares 为 JWT 角色鉴权、请求 ID、安全响应头；异常返回可恢复错误。
- backend/scripts/seed_demo.py 导入数据；migrations/001_initial.sql 说明建表策略，当前由 SQLAlchemy create_all 建表，尚未采用版本迁移工具。
- frontend/src/api、stores、types、components/common、hooks、pages、router、utils、constants 为请求、状态、契约、共享表格/标签/字段、表单、五页面与路由。
- 共享枚举 ROLES、STATES、ZONES：backend/src/constants/enums.py 与 frontend/src/constants/enums.js；校验在 backend/src/schemas/business.py，任务状态在 worker.py，UI 标签在 components/common/status.js。
- backend/src/services/modeling.py 为切分、训练、基线与解释；modeling_service.py 为训练/发布/回滚/预测编排；artifacts.py 为原子写文件。

本地开发：安装 backend/requirements.lock（真实运行环境完整依赖锁定；requirements.txt 为直接依赖）；启动本地 MySQL/Redis，设置 DATABASE_URL/REDIS_URL/JWT_SECRET 后 `uvicorn src.main:app --reload`。前端可由 Nginx 提供，代理 `/api/` 去除前缀转发给后端。

环境变量：COMPOSE_PROJECT_NAME（容器前缀）、FRONTEND_PORT/BACKEND_PORT/DB_PORT（映射）、DB_PASSWORD（演示数据库密码）、JWT_SECRET、SEED_DEMO。仅内部服务接触 Redis，不映射宿主端口。编辑班次与样本 API 支持 PUT；UI 当前以新增和浏览为主，无批量 CSV 导入、删除、分页或邮件通知。Google Fonts 需联网，无法联网则浏览器回退 serif/sans-serif。证据和测试在批次根 test-artifacts/gb-3102，不写入源码。

License: MIT。

NETWORK_SUBNET 默认 10.253.102.0/24，适配本机默认地址池耗尽的情况；迁移到其他环境可修改为空闲子网。
