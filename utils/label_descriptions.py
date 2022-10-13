std_label_table = {
    'shipping': 0,
    'product': 1,
    'processing': 2,
    'etc': 3,
}

std_three_label_table = {
    'shipping': 0,
    'product': 1,
    'processing': 2,
}

efl_category_label_descriptions = {
    'shipping': '배송과 관계 있는 문장이다',
    'product': '제품과 관계 있는 문장이다',
    'processing': '처리와 관계 있는 문장이다',
    'etc': '배송, 제품, 처리와 관계가 없는 문장이다',
}

efl_three_category_label_descriptions = {
    'shipping': '배송과 관계 있는 문장이다',
    'product': '제품과 관계 있는 문장이다',
    'processing': '처리와 관계 있는 문장이다',
}

scl_label_table = {
    'shipping': {
        'shipping': 0,
        'product': 1,
        'processing': 2,
        'etc': 3,
    },
    'product': {
        'shipping': 4,
        'product': 5,
        'processing': 6,
        'etc': 7,
    },
    'processing': {
        'shipping': 8,
        'product': 9,
        'processing': 10,
        'etc': 11,
    },
    'etc': {
        'shipping': 12,
        'product': 13,
        'processing': 14,
        'etc': 15,
    },
}

scl_three_label_table = {
    'shipping': {
        'shipping': 0,
        'product': 1,
        'processing': 2,
    },
    'product': {
        'shipping': 3,
        'product': 4,
        'processing': 5,
    },
    'processing': {
        'shipping': 6,
        'product': 7,
        'processing': 8,
    },
}

std_sentiment_label_table = {
    'negative': 0,
    'positive': 1
}

efl_sentiment_label_descriptions = {
    'negative': '불만의 감정 이다',
    'positive': '불만이 아닌 감정 이다',
}

sentiment_scl_label_table = {
    'negative': {
        'negative': 0,
        'positive': 1
    },
    'positive': {
        'negative': 2,
        'positive': 3
    }
}
