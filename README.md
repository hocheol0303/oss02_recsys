이 README 파일이 있는 위치에서 model*/train.py를 실행하는 것을 기준으로 베이스라인 작성했음

패키지 인식을 위해 root에서 pip install -e . 코드 실행 - ~~~.egg-info 폴더 생성될 것

baseline directory 구조(model03_id_NCF만 작성)
.
├── README.md
├── data
│   ├── rating_test.csv
│   └── rating_train.csv
├── model01_content_based
│   └── __init__.py
├── model02_yelp_NCF
│   └── __init__.py
├── model03_id_NCF
│   ├── __init__.py
│   ├── dataloader03.py
│   ├── inference03.py
│   ├── model03.py
│   ├── saved_models
│   └── train03.py
├── model04_demographic_filtering
│   └── __init__.py
├── setup.py
├── tmp
│   ├── inference_result.csv
│   ├── ncf
│   │   ├── movie_info.ipynb
│   │   ├── movie_split.py
│   │   ├── ncf-test.ipynb
│   │   ├── origin
│   │   │   ├── genome_scores.csv
│   │   │   ├── genome_tags.csv
│   │   │   ├── link.csv
│   │   │   ├── movie.csv
│   │   │   ├── rating.csv
│   │   │   └── tag.csv
│   │   ├── rating_test.csv
│   │   └── rating_train.csv
│   ├── wandb_key.txt
│   └── yelp.ipynb
├── utils.py
├── wandb
└── yelp
    ├── yelp_academic_dataset_business.json
    ├── yelp_academic_dataset_checkin.json
    ├── yelp_academic_dataset_review.json
    ├── yelp_academic_dataset_tip.json
    └── yelp_academic_dataset_user.json

train.py의 wandb_key에 wandb 사이트 우측 상단 - user settings - API keys의 내용을 .txt 파일로 저장하고 그 경로를 입력하면 로깅 가능. 이 때, github에 올라가지 않도록 tmp 폴더에 넣어야 함.(어떤 사이트인지간에 API key는 무조건 비공개!!)

tmp_movie_data 브랜치에서 model03_id_NCF/*.csv를 볼 수 있음.
경로 설정, wandb key 설정 후 root 디렉토리에서 python model03_id_NCF/train03.py 코드 실행하면 실행 가능해야함!!