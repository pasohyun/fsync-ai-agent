# src/intent_extraction — 12라벨 기준 의도 추출

## 목적
컴백 기획안/발매보고서 PDF에서 **기획사가 의도한 반응**을, 유튜브 댓글에 적용된 것과
**동일한 GUIDE.md 12라벨 정의**로 추출. 매핑 테이블 없이 의도 벡터 ↔ 반응 벡터를 직접
코사인 비교(Gap Score)하기 위함.

## 산출물

| 파일 | 내용 |
|---|---|
| `intent_oktapbang.json` / `_negane` / `_everlasting` / `_hwanjeolgi` | 릴리즈별 최종 의도 (2차 검증 반영본). 12라벨 `intended_label_profile` + `extra_intent_outside_12label` + `page_classification` + `extraction_audit` |
| `intent_matrix.md` | 12라벨 × 4릴리즈 매트릭스 + Gap Score용 수치 벡터(11차원) + extra_intent 목록 |
| `_audit_<release>.json` | 2차 검증 `change_log` + 1차 추출 원본 (감사 추적용) |
| `GUIDE.md` | (입력) 반응 축 12라벨 정의 |

## 방법 (1차 추출 → 2차 검증 2단 분리)

1. **페이지 분류**: 각 페이지를 [기획]/[사후]로 먼저 분류해 누수 차단. [사후](발매 후 성과·반응 통계)는 의도 근거에서 완전 제외.
2. **1차 추출**: [기획] 페이지만 근거로 11개 라벨에 level(high/medium/low/none)+rationale+evidence(원문 인용) 부여. 기타_노이즈는 항상 `not_applicable`.
3. **2차 검증**: 별도 에이전트가 evidence.quote를 원본 페이지와 한 건씩 대조 → 허위 인용·사후 누수·확증편향 정정. 모든 변경은 `change_log`에 기록.

## 검증에서 잡힌 주요 정정 (확증편향·누수 차단)

- **옥탑방**: 1차의 신규_유입·역주행_기대 근거('재미요소 가미')는 인터뷰 콘텐츠 내부 규칙일 뿐 타겟팅 전략이 아니라 → none. 장기_팬덤은 페이지 헤더가 '사후 프로모션'인 페이지 출처라 누수 → none. kihoek p4·p5를 사후로 재분류.
- **Everlasting**: 1차 이별_감성이 **텍스트 없는 컨셉포토 표지 페이지를 인용한 허위 인용** → none. 인용 오타('흐러도'→'흘러도') 교정.
- **환절기**: 음악성·청량_여름·역주행_기대를 medium→low로 하향(밴드 사운드 카타르시스를 청량으로 과대 귀속한 것 등). 신규_유입 보조 근거('설렘'을 '셀럽'으로 오독한 참고곡 문구) 제외.
- **네가내맘에**: (change_log 참조)

## 라벨 부여 원칙 (GUIDE.md와 동일한 절제)
- 명시적 근거 문장이 있어야 부여. none이 정상적 결과.
- 더 구체적인 라벨 우선(음악성보다 밴드_정체성/연주_악기/보컬_라이브).
- 음악성·신규_유입·역주행_기대는 의도 축에서 약하거나 비어도 정상 — 억지로 채우지 않음.
- 12라벨로 설명 안 되는 의도(상징·해석, 티징·기대감, 유머)는 `extra_intent_outside_12label`에 별도 기록 → Gap 리포트의 "측정 불가 영역" 정성 발견으로 활용.

## 다음 단계 (Gap Score 계산, 별도 작업)
`intended_label_profile`만 벡터화(level→{none:0,low:1,medium:2,high:3}) → 댓글 12라벨 분포 벡터와 코사인 비교.
`extra_intent_outside_12label`은 수치 비교에서 제외하되 "회사가 의도했으나 반응 축으로 측정 불가능한 영역"으로 리포트에 포함.
