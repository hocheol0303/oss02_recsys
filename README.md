이 README 파일이 있는 위치에서 model*/train.py를 실행하는 것을 기준으로 베이스라인 작성했음 <br>
<br>
패키지 인식을 위해 root에서 pip install -e . 코드 실행 - ~~~.egg-info 폴더 생성될 것 <br>
<br>
baseline directory 구조(model03_id_NCF만 작성) <br>
. <br>
├── README.md <br>
├── data <br>
│   ├── rating_test.csv <br>
│   └── rating_train.csv <br>
├── model01_content_based <br>
│   └── __init__.py <br>
├── model02_yelp_NCF <br>
│   └── __init__.py <br>
├── model03_id_NCF <br>
│   ├── __init__.py <br>
│   ├── dataloader03.py <br>
│   ├── inference03.py <br>
│   ├── model03.py <br>
│   ├── saved_models <br>
│   └── train03.py <br>
├── model04_demographic_filtering <br>
│   └── __init__.py <br>
├── setup.py <br>
├── tmp <br>
│   ├── inference_result.csv <br>
│   ├── ncf <br>
│   │   ├── movie_info.ipynb <br>
│   │   ├── movie_split.py <br>
│   │   ├── ncf-test.ipynb <br>
│   │   ├── origin <br>
│   │   │   ├── genome_scores.csv <br>
│   │   │   ├── genome_tags.csv <br>
│   │   │   ├── link.csv <br>
│   │   │   ├── movie.csv <br>
│   │   │   ├── rating.csv <br>
│   │   │   └── tag.csv <br>
│   │   ├── rating_test.csv <br>
│   │   └── rating_train.csv <br>
│   ├── wandb_key.txt <br>
│   └── yelp.ipynb <br>
├── utils.py <br>
├── wandb <br>
└── yelp <br>
    ├── yelp_academic_dataset_business.json <br>
    ├── yelp_academic_dataset_checkin.json <br>
    ├── yelp_academic_dataset_review.json <br>
    ├── yelp_academic_dataset_tip.json <br>
    └── yelp_academic_dataset_user.json <br>
<br>
train.py의 wandb_key에 wandb 사이트 우측 상단 - user settings - API keys의 내용을 .txt 파일로 저장하고 그 경로를 입력하면 로깅 가능. 이 때, github에 올라가지 않도록 tmp 폴더에 넣어야 함.(어떤 사이트인지간에 API key는 무조건 비공개!!) <br>
<br>
tmp_movie_data 브랜치에서 model03_id_NCF/*.csv를 볼 수 있음. <br>
경로 설정, wandb key 설정 후 root 디렉토리에서 python model03_id_NCF/train03.py 코드 실행하면 실행 가능해야함!! <br>