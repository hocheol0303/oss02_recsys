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