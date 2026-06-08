# Phase 7 — Infra runbook (smoke-verified local, KHÔNG trong CI)

> Theo quyết định P7 (spec §3): infra nặng (server thật) dựng + **smoke thủ công ở local**, không nhồi vào GitHub Actions — đúng tiền lệ Docker P5. Số liệu định lượng (DORA, overhead) đo ở P8.

## 1. MLflow server thật (Exp-3, framework được chọn)

Server MLflow nhẹ (1 process) làm backend tracking + registry thật để **đo overhead/RSS/governance** ở Exp-3 (P8) và để serving nạp model qua registry thật.

```bash
# Khởi động
docker compose -f docker-compose.mlflow.yml up -d
# Smoke: server healthy
curl -sf http://127.0.0.1:5000/health        # -> OK
# UI: http://127.0.0.1:5000  (experiments + Models registry)
# Dùng từ pipeline: trỏ MLFLOW_TRACKING_URI vào server thật thay cho sqlite
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m src.training train --dataset artifacts/cicd/fixture.parquet \
  --output-dir artifacts/cicd/models --backend mlflow
# Dừng
docker compose -f docker-compose.mlflow.yml down
```

Backend store + artifacts nằm ở volume `./artifacts/mlflow_server/` (gitignored).

## 2. ClearML — finding vận hành (biện minh chọn MLflow, Q2)

ClearML là **framework phụ** (trục so sánh Exp-3, vai trò "biện minh chọn MLflow"). Trong luận văn, overhead ClearML đo ở **offline mode** (library-level) — xem unit test `tests/tracking/test_clearml_tracker.py`.

**KHÔNG dựng ClearML self-host stack.** Đây là một *finding vận hành* đáng ghi nhận:

| Tiêu chí vận hành | MLflow (được chọn) | ClearML self-host |
|---|---|---|
| Thành phần để chạy server | **1 process** (`mlflow server` + backend DB + artifact dir) | **~5 container**: mongodb + elasticsearch + redis + apiserver + webserver + fileserver |
| Tài nguyên (single-node NWDAF) | nhẹ | nặng (elasticsearch + mongo đói RAM) |
| Phù hợp phạm vi luận văn (single-node, tập trung) | ✅ | ❌ rủi ro lệch sang DevOps hạ tầng (CLAUDE.md §3.7) |

→ Với bối cảnh NWDAF single-node của luận văn, **độ phức tạp vận hành** nghiêng hẳn về MLflow. Đây là một phần lý do chọn MLflow cho C5/C6 (cùng với các tiêu chí Exp-3 khác).

**Threat-to-validity (ghi ở spec §10):** ClearML offline không đo được server-RSS head-to-head với MLflow server thật → overhead ClearML là library-level; bù lại bằng finding vận hành định tính ở trên. Đây là lựa chọn phạm vi có chủ ý (trục phụ), nêu rõ ở phần giới hạn.
