# AutoStock — KOSPI+KOSDAQ Daily Scanner

매일 **한국시간 오전 10시**에 코스피+코스닥 전체(ETF/ETN/스팩 등 이름기반 제외)를 스캔하여
래리 윌리엄스 지표(%R), RSI, OBV, 20일선, 음봉/양봉 조합으로 매수/매도/홀드 신호를 산출하고
**Gmail**로 HTML 리포트를 전송합니다.

## 설치 & 실행
1. 이 레포를 GitHub에 업로드
2. 레포 **Settings → Secrets → Actions** 에 `GMAIL_APP_PASSWORD` 등록 (구글 앱 비밀번호)
3. `config.toml`의 발신/수신 메일주소 수정
4. (옵션) `universe.max_symbols`로 테스트 범위 제한 후 정상 동작 확인
5. 워크플로우는 기본적으로 매일 KST 10:00에 자동 실행 (수동 실행도 가능)

## 주요 파일
- `main.py` : 엔트리포인트
- `data_sources.py` : pykrx/yfinance 데이터 로더, 코스피+코스닥 전체 티커 로더
- `indicators.py` : %R, RSI, OBV, 이동평균, 캔들, 기울기
- `strategy.py` : 신호 생성 로직
- `reporter.py` : HTML 리포트 생성
- `mailer.py` : Gmail SMTP 발송
- `.github/workflows/run.yml` : GitHub Actions 스케줄러 (UTC 01:00 = KST 10:00)

## 설정 (`config.toml`)
- `[universe] krx_mode="ALL"` → 코스피+코스닥 전체 자동 로드
- `exclude_etf_etn_spac=true` → 간단한 이름기반 필터
- `max_symbols=0` → 무제한 (성능상 처음엔 100~300 정도로 테스트 권장)

## 주의
- Gmail은 **앱 비밀번호** 필수
- 대량 종목 스캔 시 실행시간 증가 가능 → `max_symbols`로 제한 후 점진 확대 권장
