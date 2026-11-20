# SOLO 流式字符编码服务与原件保真修订库

本题面定义的 SOLO 编码（仅实现题面格式，不接入任何外部编码标准）的 Python 3.11
实现，以及在其之上构建的、按字符区间打补丁但保持未修改原始字节的修订库。

## 交付物

| 路径 | 内容 |
| --- | --- |
| `solo/codec.py` | 流式解码器 `Decoder`、检查点 `Checkpoint`、整件解码 `decode_document` / `decode_all` / `iter_decode`、独立编码入口 `encode_text` / `encode_document`，错误类型与路径记录 |
| `solo/revision.py` | 修订库 `RevisionLibrary`、`replace` / `preserve` 补丁、版本与批次记录、边界模式调整、字节保真校验 |
| `solo/timing.py` | 分段成绩档案：可复跑行格式解析（含各数字的原件字符/字节位置）、整场解释位集 DP、官方段全部可行耗时与端值见证全程、最早无合法前缀的行、记录版本谱系、独立穷举校验入口 |
| `solo/relay.py` | 接力报名：独立保存的报名表解析、最坏情形保证下的阵容选择、最小最坏总时与并列阵容、必定上场队员、无合法阵容与超标的区分 |
| `solo/swimdemo.py` | 游泳演示：真实编码原件、申报更正与裁判勘误两批修订、只变报名表的对照、一人备选两棒与平手阵容小例、小样穷举验证，写出 `artifacts/swim_*` |
| `solo/sample.py` | 确定性合成原件生成器（ESC/PAIR 故意跨 128 KiB 块，含模式切换、空替换、连续补丁与非法尾部所需素材） |
| `solo/demo.py` | 演示：真实原件、两批修订、未修改字节核对、错误报告、较大语料，写出 `artifacts/` |
| `solo/__main__.py` | CLI：`decode/check/encode/revise/demo` 与 `timing/relay/swimdemo` |
| `tests/test_solo.py` | 独立参考状态机、差分测试、跨块、检查点、修订、规模测试 |
| `tests/test_swim.py` | 题面例子逐项、陷阱段、矛盾行、位置与谱系、随机小样 DP↔穷举差分、阵容选择差分、限额与错误 |
| `artifacts/` | 运行 `python3 -m solo.demo` 与 `python3 -m solo.swimdemo` 生成的真实原件、各版本字节与清单 |

## SOLO 编码（题面定义）

- 正文从 `SINGLE` 模式开始。
- `SINGLE`：`00..7F` → 同码点 ASCII；`FF` 为 ESC；`80..FE` 非法。
- ESC 后单字节指令：`01` → 切到 `PAIR`；`00` → 切回 `SINGLE`；`FF` → 字面
  `U+00FF` 且不改模式；其它 → 错误。模式切换本身不产字符。
- `PAIR`：除 `FF`（仍是 ESC）外，两个字节组成无符号大端整数 `n`，映射
  `U+4E00 + n`，限定 `0 ≤ n ≤ 0x0FFF`。第一字节正常时，第二字节即使是 `FF`
  也属于该数值（`00 FF` = n=0x00FF），不会重新开转义。
- 正文结束遇到半个 PAIR 或半条 ESC 均为错误。
- 错误报告原字节偏移与到达该状态的路径（`DecodeError.offset` / `.path`），绝不
  用 `?` 替换后继续。解码失败后解码器进入失败态，不会假成功。

## 流式语义

`Decoder` 按任意字节块增量运行：`feed()` / `feed_into()`，偏移始终是原件正文
偏移。跨块半条 ESC、半个 PAIR 会被保存为待续状态，下一块首字节续接，因此不会
像旧的按块重新初始化解码器那样丢掉块首字符。`finish()` 校验正文末尾并返回结束
模式。

### 检查点

`Decoder.checkpoint()` 返回 `Checkpoint`，包含位置、模式、待续状态与**前缀
SHA-256**；`Checkpoint.resume(body)` / `Decoder.from_checkpoint()` 只在给定正文
前缀摘要一致时才从检查点继续（`verify=False` 可跳过，供已自行校验前缀的调用方
使用）。`Version.checkpoint_at(byte_offset)` 可直接从某个版本继续解码。

## 修订库

- 版本是不可变的真实原件字节 + 有序补丁批次；一个库先经 `add_original()` 载入
  原件（会完整解码校验，失败即拒绝）。
- `replace(start_char, end_char, text)`：按某固定原件版本的**字符**半开区间替换
  为 Unicode 文本。`preserve(start_byte, end_byte)`：要求该**字节**区间原样保留
  （可以落在字符内部；包含它的字符因此不可被替换；若边界落在纯切换字节上则拒绝）。
- 一批补丁先校验：同一基准版本、替换区间两两不相交（相邻允许）、preserve 区间
  互不重叠且不被替换触碰、替换文本可编码；任一失败则整批不发布。
- 组装：未替换字符按原字节复制（`kept_runs` 记录旧字符区间与新字节区间的对应），
  只对替换文本编码；当前输出模式与下一段原始字节解码模式不一致时，才在拼接边界
  插入合法切换（`BoundaryAdjustment`，`splice` / `boundary` / `final`）。整文件
  从不重新编码，因此不会掩盖原件差异。
- 新版本结束模式强制为 `SINGLE`，不会给后续保留字节留下悬空模式。
- 发布前独立校验：重新流式解码新正文，逐段核对未替换区间字节一致、preserve
  区间逐字节一致，并核对解码文本等于“旧 Unicode 做同批替换”的结果。全部通过
  才登记新版本；否则抛出 `RevisionError`，库不变化。

### 版本记录

`Version.records` / `BatchRecord` 记录每个替换的旧字符区间、旧字节区间、新字符
区间、新字节区间、新增字节与边界切换；`boundary_adjustments` 记录新字节偏移、
前后模式和切换字节；`kept_runs` 与 `dropped_switch_spans` 记录原样复制和未保留的
纯切换跨度。

## 分段成绩档案（`solo/timing.py`）

旧电子计时牌只显示累计计时对 100 取余的两位读数；同一个 `05` 可能已越过一圈或
两圈。档案是 SOLO 正文里的可复跑行格式（`#` 起注释行）：

```
SESSION <编号> <队员> <泳姿>
TOUCH <rr> <dlo> <dhi>                  两位读数；与上次触壁的允许耗时闭区间
TOUCH <rr> <dlo> <dhi> REF <alo> <ahi>  另有独立裁判给的绝对累计闭区间
OFFICIAL <a> <b>                        官方取段：第 a 次到第 b 次触壁（a=0 为出发）
END
```

- 出发时绝对累计为零；最多 20 次触壁，绝对累计不超过 2000；只保留从出发到最后
  一触同时满足所有记录的整场解释。
- 解析交付每条记录所在行、每个数字的原件字符区间与字节区间（沿用公开的分块解码
  与字节映射）；`format_archive` 可规范化重放。
- 分析对每次触壁求可达绝对累计集合（读数 mod 100、允许区间、裁判区间同时满足），
  官方段耗时取所有全程解释的 `t_b−t_a`，给出全部可行耗时，并为最早、最晚各给
  一条完整全程与逐触壁计算依据。两端各自取最早最晚再相减可能拼出从未发生的一段
  （主例中 t1∈{5,105}、t2∈{105,205}，朴素相减得 [0,200]，真实取值只有 100），
  报告会显式标出该陷阱。
- 记录相互矛盾时，指出最早无合法前缀的触壁行及其依据。
- 裁判勘误等更正走修订库批次；`char_lineage` 把一条记录沿版本链回溯到引入它的
  批次，历史版本可重新分析复核，互不改写。

## 接力报名（`solo/relay.py`）

报名表独立保存（改变报名资格不改写成绩档案），行格式：

```
TARGET <目标总时>
LEG <1..4> <泳姿>
ENTER <队员> <训练编号>
```

- 每棒给出泳姿、可报名队员与指定训练编号；每人整队最多上一棒；不同队员的训练
  相互独立，整队最坏总时为各棒最坏耗时之和；最多 12 名队员。
- 只用四棒均有有效训练的阵容（训练存在、属于该队员、泳姿相符、有合法计时解释；
  无效条目逐条给出排除原因）。
- 先保证所有合法计时解释下整队不慢于目标，再在达标阵容中取最小最坏总时；所有
  并列阵容中必定上场的队员一并给出。无合法阵容与有阵容但全部超标明确区分，后者
  给出最接近的最坏总时。

## 运行

```bash
python3 -m unittest discover -s tests -v      # 全部测试
python3 -m solo.demo                            # 编码/修订演示并写 artifacts/
python3 -m solo.swimdemo                      # 游泳接力演示并写 artifacts/swim_*
python3 -m solo check artifacts/original.bin
python3 -m solo decode --checkpoint 10 artifacts/version_1.bin
python3 -m solo revise artifacts/original.bin patches.json > new.bin
python3 -m solo timing artifacts/swim_archive_v0.bin
python3 -m solo relay artifacts/swim_archive_v0.bin artifacts/swim_entry_main.bin
```

`patches.json` 格式：

```json
{"patches": [
  {"replace": [0, 3, "新文本"]},
  {"preserve": [100, 200]}
]}
```

## 测试覆盖

- 1/2/7/4096 字节分块对独立朴素状态机逐字符、逐字节区间核对（含非法输入与
  错误偏移）。
- 生成器把 ESC/PAIR 精确放在 128 KiB 边界上；跨块、模式切换无字符、替换为空、
  连续/相邻补丁、非法尾部均有专项测试。
- 修订：未修改区间字节等同、每版本可重开解码、失败批次不发布、preserve 逐字节
  保留、新文本等于同批替换、无悬空模式。
- 规模：10 MiB 原件、一批 100 个补丁（`ScaleTests`），未修改运行按原字节复制。
- 游泳：题面例子逐项核对（A 段只能 100、限 130 只有 A 达标、申报改 [100,200]
  后 230/132 均超标、裁判勘误后回到 100、只取消 A 资格最佳变 132）；端值相减
  陷阱；最早矛盾行；数字原件位置；版本谱系与历史复核；随机小样上位集 DP 与独立
  穷举差分、阵容选择与独立穷举差分；触壁/累计/队员限额与格式错误。
