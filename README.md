# CloudTrail 이상 탐지 — MITRE ATT&CK 기반

멀티 클라우드 로그 속에서 **실제 위협이 되는 이벤트만 자동으로 선별**하는 보안 이상
탐지 시스템입니다. **flaws.cloud AWS CloudTrail** 데이터셋(약 190만 건)을 기반으로,
쏟아지는 클라우드 API 호출 중 공격 활동을 자동으로 찾아냅니다.

> 아주대학교 프로젝트 · 2025.03 – 2025.06 · Python / pandas / scikit-learn / PyTorch

## 문제 정의와 접근

CloudTrail은 모든 AWS API 호출을 기록합니다. 대부분은 정상 자동화 트래픽이고,
극히 일부가 공격자의 활동(정찰 → 자격 증명 접근 → 지속성 확보 → 권한 상승 →
방어 회피 → 유출/파괴)입니다. 데이터셋에는 **라벨이 없어서** 다음과 같이 진행했습니다.

1. **MITRE ATT&CK 기반 약(weak) 라벨링** — 각 API 호출을 해당하는 ATT&CK 택틱에
   매핑하고, 공격 성격의 택틱(+ 권한 오류 같은 리스크 신호)을 *공격*으로 표시.
   결과는 약 **정상 57% / 공격 43%**.
2. **지도학습 baseline → 실패 진단.** `eventName` 원-핫 피처로 RandomForest를 학습하면
   랜덤 분할에서 F1 **약 0.99**가 나오지만, 이는 **라벨 누수**입니다. 라벨 자체가
   `eventName`에서 파생됐기 때문에 모델이 라벨을 정의하는 컬럼을 그대로 **암기**하는
   것뿐입니다. **leave-one-tactic-out**(한 택틱을 통째로 학습에서 제외) 평가에서는
   미지의 택틱(자격 증명 접근, 유출)에 대한 재현율이 **~0.22**로 붕괴합니다. 모델은
   "악성"을 학습한 게 아니라 API 이름을 외운 것입니다.
3. **비지도 이상 탐지로 전환.** 학습 시 라벨을 쓰지 않고, 라벨은 평가에만 사용합니다.
   네 가지 탐지기를 비교했습니다.
   - **Isolation Forest** (행위 기반 피처)
   - **Local Outlier Factor (LOF)**
   - **오토인코더** (PyTorch, 재구성 오차) — torch가 없으면 PCA 폴백
   - **MiniLM 임베딩 + Isolation Forest** (이벤트의 의미적 표현)

## 결과 (합성 샘플 기준; 전체 데이터는 다운로드 스크립트로)

```
모델                       pr_auc   roc_auc   recall@thr   best_f1
isolation_forest           0.63     0.77       0.66        0.70
autoencoder                0.60     0.75       0.60        0.75
minilm_iforest             0.59     0.58       0.44        0.59
lof                        0.49     0.57       0.52        0.59
```

지도학습: 랜덤 분할 F1 **0.99** vs leave-one-tactic-out 평균 재현율 **~0.75**
(자격 증명 접근·유출 같은 미지 택틱에서는 **~0.22**로 붕괴) — 비지도 전환을 결정하게 된
핵심 발견입니다.

**스트리밍:** 학습된 Isolation Forest로 샘플을 실시간 스코어링하고 상위 이상치를
선별하면, MITRE 약라벨 대비 **약 90% 정밀도**로 위협을 표면화합니다. 즉 대응 담당자가
보는 알림이 실제로 유의미합니다.

## 저장소 구조

```
src/
  data/       CloudTrail JSON 파싱 · 피처 엔지니어링 · Kaggle 다운로드
  labeling/   MITRE ATT&CK 택틱 매핑 + 약 라벨링
  models/     지도학습 · isolation_forest · lof · autoencoder · minilm+iforest · 평가
  serving/    온라인 스코어러 + 실시간 이벤트 스코어링 FastAPI 서비스
  pipeline.py 학습 → 비교 → 최적 탐지기 저장 (end-to-end)
notebooks/    01 EDA · 02 MITRE 라벨링 · 03 지도학습 baseline · 04 비지도 비교
scripts/      make_sample · replay_stream
```

## 빠른 시작

```bash
pip install -r requirements.txt

python scripts/make_sample.py        # 실행 가능한 CloudTrail 샘플 생성
python -m src.pipeline               # 라벨링 → 지도학습 진단 → 모델 비교 → 최적 모델 저장
```

이후 `notebooks/`에서 전체 분석 과정을 확인하거나, 학습된 탐지기를 서빙할 수 있습니다.

```bash
uvicorn src.serving.app:app          # 스코어링 API + 실시간 뷰 http://localhost:8000
python scripts/replay_stream.py      # 샘플을 실시간 스트림으로 리플레이
```

## 전체 데이터셋

```bash
pip install kaggle                   # ~/.kaggle/kaggle.json 필요
python -m src.data.download          # nobukim/aws-cloudtrails-dataset-from-flaws-cloud
python -m src.pipeline data/raw      # 전체 약 190만 건으로 실행
```

## 기술 스택

Python · pandas · scikit-learn (IsolationForest / LOF / RandomForest) · PyTorch
(오토인코더) · sentence-transformers (MiniLM) · FastAPI · MITRE ATT&CK
