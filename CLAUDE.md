# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Ngôn ngữ làm việc:** Tiếng Việt cho mọi prose (trả lời, docs, luận văn). Code identifiers/docstrings bằng tiếng Anh. Thuật ngữ kỹ thuật giữ song ngữ Việt–Anh.

---

## 1. Repo này là gì

Workspace cho **luận văn thạc sĩ (UET-VNU)** của **Mai Xuân Ngọc** (MSHV 24025038), ngành **Kỹ thuật Phần mềm**. HD: PGS.TS Võ Đình Hiếu; đồng HD: TS. Đỗ Văn Chiểu.

- **Tên đề tài:** *Nghiên cứu và xây dựng quy trình MLOps trong hệ thống phân tích dữ liệu mạng lõi di động thế hệ thứ 5.*
- **Hướng (ĐÃ CHỐT, thầy Hiếu duyệt):** **Ứng dụng** — *thật sự xây dựng & vận hành* một pipeline MLOps end-to-end cho bài toán 5G NWDAF, rồi **đo lường có đối chứng** giá trị nó mang lại. KHÔNG phải xây "khung đánh giá MLOps tổng quát".
- **Use case khóa cứng:** Phát hiện bất thường **ping-pong handover** của UE (thuê bao chuyển giao qua lại liên tục giữa 2 cell) từ dữ liệu **EBS** (Event-Based Statistics) của NWDAF.
- **Trạng thái:** Workspace **mới**. Chủ trương: **build fresh theo `docs/THESIS_SPEC.md`**, chỉ *tái dùng có chọn lọc* logic đã kiểm chứng từ repo cũ (xem §8). Mục §5 mô tả **kiến trúc MỤC TIÊU**.
- **Thời gian:** **KHÔNG còn ràng buộc cứng** — ưu tiên *hoàn chỉnh & đúng* hơn là cắt giảm cho kịp. Git đã `init` + add remote.
- **LSTM-AE dùng PyTorch** (chạy trên Python 3.14, một venv duy nhất) — `pip install -e ".[lstm]"` cài `torch>=2.6`; không cần venv riêng cho TF/3.12.

---

## 2. ⚠️ Bối cảnh "quay xe" — ĐỌC TRƯỚC KHI LÀM BẤT CỨ VIỆC GÌ

Đây là **bản làm lại lần 2**. Bản seminar cũ bị hội đồng phê bình nặng; mọi quy tắc dưới đây sinh ra để **không lặp lại sai lầm đó**.

| Người phản biện | Phê bình bản cũ | Hệ quả cho repo này |
|---|---|---|
| **Thầy Trình** | "Thao tác hóa thuộc tính chất lượng **thủ công, cảm tính, không tái lập**"; baseline thô → kết luận hiển nhiên; claim "đánh giá toàn diện" là nói quá; **phải so sánh với framework chuẩn (MLflow/Kubeflow)** | Chỉ dùng **metrics đo tự động bằng code**; dùng **framework chuẩn thật**; không claim quá lời |
| **Cô Thủy** | "Tên đề tài và nội dung **không khớp**" (tên hướng 5G nhưng nội dung là benchmark MLOps chung) | Nội dung phải **thật sự giải bài toán 5G**; mọi phần map về handover/NWDAF |
| **Thầy Đức Anh** | "Trình bày **không liên quan** yêu cầu bài toán đề ra" | Luôn bám sát kết quả phát hiện ping-pong handover |

---

## 3. 🚫 Anti-patterns — TUYỆT ĐỐI TRÁNH

1. **KHÔNG chấm điểm chất lượng bằng tay/cảm tính** (vd `Traceability 0.23→0.92` tự gán). Mọi con số phải sinh ra từ code đo được, tái lập được.
2. **KHÔNG dùng baseline thô để rút "kết luận hiển nhiên"** ("có MLOps thì traceable hơn" — ai cũng biết). Baseline phải phục vụ **định lượng chi phí/lợi ích**, không phải để "thắng".
3. **KHÔNG biến luận văn thành benchmark/khung đánh giá MLOps tổng quát.** 5G/NWDAF là **động lực thiết kế**, không phải bối cảnh tùy chọn.
4. **KHÔNG claim "toàn diện"** khi chưa có cơ sở. Phát biểu đóng góp **đúng phạm vi thực tế đã làm**.
5. **KHÔNG bịa dữ liệu/kết quả.** Tuyệt đối. Chưa chạy thì ghi "chưa chạy", không điền số phỏng đoán.
6. **KHÔNG đào sâu lý thuyết viễn thông** — giữ góc nhìn **Kỹ thuật Phần mềm** (thiết kế & vận hành hệ thống ML).
7. **KHÔNG dừng ở "đồ án ráp công cụ".** Mọi implementation phải gắn với một RQ + một metric khách quan (xem §4). Re-implement tutorial + đổi dataset = báo cáo kỹ thuật, KHÔNG đủ tầm thạc sĩ.

---

## 4. Trục nghiên cứu (the research spine) — phần nâng "implementation" thành "luận văn"

Mỗi dòng code phải truy về được một trong các trục sau.

**Câu hỏi nghiên cứu (ĐÃ CHỐT — duyệt 06/2026):**
- **RQ1** — Dữ liệu NWDAF/EBS có đặc thù gì (streaming, nhãn yếu, concept drift, multi-source) đặt ra thách thức nào cho vận hành ML?
- **RQ2** — Pipeline MLOps cần thiết kế thế nào để giải quyết các thách thức đó?
- **RQ3** — MLOps mang lại giá trị đo được gì so với quy trình ad-hoc, trên 6 nhóm metrics?
- **RQ4** — Pipeline có linh hoạt khi drift / thay model / đổi framework không?

**Thiết kế thực nghiệm có đối chứng (ĐÃ CHỐT — chi tiết: `docs/research/METHODOLOGY_GROUNDING.md` §B, và `docs/THESIS_SPEC.md`):**
- **Trục 1 — Capability ablation (xương sống):** **C0 (không MLOps)** → **C1 (full MLOps trên MLflow)** cho cùng bài toán handover, cùng data/model/code. C0 là pipeline bị *gỡ* lớp MLOps, KHÔNG phải hệ tự dựng yếu → tránh phê bình "straw-man". Đo cả 6 nhóm metrics; khung theo Google maturity L0→L1(→L2).
- **Trục 2 — So sánh framework (phụ, thỏa thầy Trình):** Noop / **MLflow (chính)** / ClearML ở **lát tracking-registry** qua `BaseTracker`; đo overhead/resource/governance, ML metrics giữ trùng. Vai trò = *biện minh chọn MLflow*. KHÔNG dựng full stack trên ClearML. **Kubeflow** = literature + hướng phát triển (ĐÃ CHỐT: core single-node, không đưa K8s vào phạm vi chính).
- **Maturity đo khách quan:** **ML Test Score (Breck et al. 2017)** — 28 test × {0 / 0.5 manual / 1 automated}, lấy MIN 4 mục; triangulate Google L0–2 & Azure L0–4 dạng tiêu chí present/absent. KHÔNG tự gán điểm.
- **ATAM/utility-tree:** chỉ để *chọn* thuộc tính cần đo, KHÔNG báo cáo điểm gán tay làm kết quả.
- **Giao thức:** N≥10 runs, fixed seeds, loại warmup, **Wilcoxon signed-rank** (α=0.05), raw JSONL, báo cáo mean±std + CI.

**6 nhóm metrics (tất cả PHẢI đo tự động):**

| Nhóm | Ví dụ metric | ⚠️ Lưu ý chống "chấm tay" |
|---|---|---|
| Model Performance | Precision, Recall, F1, ROC-AUC, PR-AUC | — |
| Operational | pipeline time, inference latency, throughput, deploy time | đo bằng `time.perf_counter()` |
| Business Impact | # ping-pong phát hiện sớm, giảm signaling/chi phí ước tính | **phải neo vào đại lượng đo được**, không nói chung chung |
| Model Drift & Data Quality | PSI, KS-test, Evidently drift share, detection latency | — |
| Cost & Resource | CPU%, RAM (RSS), storage overhead | đo bằng `psutil` |
| MLOps Maturity Level | Google/Azure maturity level | **dựa trên bằng chứng có/không artifact, KHÔNG tự cho điểm** |

> Hai nhóm **Business Impact** và **Maturity** là chỗ dễ tái phạm lỗi "cảm tính" nhất — luôn dựa trên bằng chứng tự động.

---

## 5. Kiến trúc MỤC TIÊU (target stack) — chưa build xong

Theo phong cách tutorial MLOps công nghiệp (DVC + Feast + MLflow + Evidently + FastAPI + Docker + GitHub Actions), **adapt cho đặc thù 5G** (anomaly detection nhãn yếu + drift + streaming, KHÁC với bài toán tabular có nhãn của tutorial).

```
EBS files (real + synthetic)
  → Data pipeline      ingest/parse EBS → snapshot → DVC versioning (+ remote)
  → Feature pipeline   7 đặc trưng handover/(IMSI, window) + weak labels → Feast feature store
  → Training           IsolationForest (chính) / LSTM-AE PyTorch (model-swap) → MLflow tracking + registry (alias/stage)
  → Serving            FastAPI /predict + Docker (compose)
  → Monitoring         drift: PSI (tự code) + Evidently (chuẩn ngành); system metrics (Prometheus/Grafana)
  → CI/CD              GitHub Actions (+ self-hosted runner): feature → train → serve/deploy
```

**Phạm vi stack (cập nhật — không còn ràng buộc thời gian → phủ ĐỦ 8/8 thành phần Kreuzberger; chi tiết `THESIS_SPEC.md` §5 + `docs/research/METHODOLOGY_GROUNDING.md` §C):**
- **MLflow** (C5 registry + C6 metadata) · **DVC** (C3 data/model versioning EBS) · **Feast** (C3 feature store — online/offline phục vụ AnLF real-time vs MTLF batch, train/serve consistency) · **FastAPI + Docker** (C7 serving) · **Evidently + Prometheus/Grafana** (C8 monitoring) · **GitHub Actions** (C1 CI/CD).
- **C2 orchestration:** DVC pipelines (`dvc.yaml` DAG) + GitHub Actions; nâng cấp orchestrator chuyên dụng nếu cần.
- **Kubeflow/Kubernetes (ĐÃ CHỐT: A):** literature + hướng phát triển; KHÔNG đưa K8s vào core.
- **Drift:** PSI 3 tầng + KS qua Evidently (nhãn thưa → KHÔNG dùng detector có giám sát). **Eval:** ROC-AUC + PR-AUC; F1 chỉ ở ngưỡng cố định, có biện minh.

**Nguyên tắc thiết kế (giữ từ bản tốt của repo cũ):**
- **Backend-agnostic:** logic pipeline đi qua một lớp trừu tượng tracker (`BaseTracker` → Noop/MLflow/ClearML), KHÔNG gọi thẳng API framework trong core. Đây là điều kiện để so sánh công bằng (biến kiểm soát duy nhất = hệ thống tracking).
- Mỗi subpackage có `schema.py` (config dataclasses) + một CLI module (`python -m <package>`).
- **Synthetic data** calibrate từ EBS thực (3 file, ~233K events, 24K handovers) → `profile.json` → generator; dùng cho kịch bản drift. Luôn nêu rõ "synthetic, calibrate từ thực".

---

## 6. Quy tắc khi đóng vai trò hỗ trợ luận văn

- **Vai trò:** cố vấn (advisor), KHÔNG phải người chấm. Phân tích, gợi mở, chỉ bước tiếp theo cụ thể.
- **KHÔNG ghost-write** nguyên chương. Khi đưa câu chữ mẫu, ghi rõ **"(ví dụ minh họa)"**.
- **KHÔNG bịa số liệu, trích dẫn, hay kết quả.** Thiếu thì nói thiếu.
- Văn phong luận văn: ngôi thứ ba khách quan, theo `vietnamese_thesis_style_guide.md` (migrate từ repo cũ).
- LaTeX template UET-VNU sẽ migrate từ `Template_thesis_master/` repo cũ.

---

## 7. Commands (toolchain dự kiến — cập nhật khi component thật sự có)

> Workspace còn trống. Khi thêm package, bổ sung lệnh thật vào đây. Dự kiến chạy trong venv, từ repo root.

```bash
# Pipeline & experiments (mẫu, theo convention python -m)
python -m src.<package> ...                 # mỗi stage là một CLI module
python -m src.experiments.cli ...           # entry point thực nghiệm (B0/B1, N runs)

# MLOps services (dự kiến)
mlflow server --backend-store-uri ... --default-artifact-root ... --port 5000
dvc init && dvc add <data> && dvc push      # data versioning
docker compose up ...                       # serving API / monitoring stack
# GitHub Actions: .github/workflows/*.yml    # CI/CD
```

**Lưu ý:** repo này KHÔNG có pytest suite kiểu truyền thống — **thực nghiệm chính là kiểm chứng**. Chạy experiment để verify thay đổi.

---

## 8. Nguồn tham chiếu (repo cũ — đọc, đừng copy mù)

Repo cũ ở `/Users/ngocmx/Thạc Sĩ/MLOps_Project/`:
- `src/` — code lõi canonical (~8.3K dòng): ingestion, features, training (IForest+LSTM), serving (FastAPI), monitoring (PSI), tracking abstraction (Noop/ClearML/MLflow), experiments. **Build fresh + tái dùng có chọn lọc** logic đã kiểm chứng (parser, feature compute, weak labels, PSI, IForest/LSTM, mẫu `BaseTracker`) — re-implement sạch, không bê nguyên.
- `docs/MASTER_THESIS_BACKBONE.md` — north star bản refactor #1 (RQ, anti-patterns).
- `vietnamese_thesis_style_guide.md` — văn phong luận văn.
- `Template_thesis_master/` — LaTeX template UET-VNU.
- `nwdaf_mlops/` — prototype ClearML legacy. **KHÔNG dùng, KHÔNG copy.**

> Bản tóm tắt `docs/TOM_TAT_LUAN_VAN.md` của repo cũ là **refactor #1** (trọng tâm so sánh framework ClearML↔MLflow). Repo Ver2 này là **refactor #2** (trọng tâm ứng dụng + đối chứng ad-hoc). Khi tham chiếu, ưu tiên hướng ứng dụng.
