# deepspeed와 transformers를 활용하여 모델을 학습하는 예시

### setup tips
* apt 또는 yum을 사용하여, `numactl`을 설치해주자.
    *  리눅스에서 프로세스나 공유메모리의 NUMA 정책을 control하는 도구다. deepspeed가 이거를 내부에서 활용하는 느낌.
* docker 컨테이너 내에서 deepspeed를 활용할 경우, 반드시 `--privileged` 옵션을 추가하여 도커를 띄우자.
    * 시스템 내 장치 등 주요 자원에 접근 가능케 하기 위함.
