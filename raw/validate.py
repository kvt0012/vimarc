import json
import os


def check_answer(answer):
    number = answer["number"]
    date = answer["date"]
    spans = answer["spans"]
    assert type(number) == str
    assert type(spans) == str
    assert type(date) == dict
    assert "day" in date
    assert "month" in date
    assert "year" in date
    assert type(date["day"]) == str
    assert type(date["month"]) == str
    assert type(date["year"]) == str
    assert len(number) or len(spans) or len(date["day"] + date["month"] + date["year"])


def validate_dataset(file_path='./vimarc.json'):
    with open(file_path, 'r') as j:
        dataset = json.loads(j.read())
    kept_count, skip_count = 0, 0
    context_id_set = set()
    question_id_set = set()
    for context_id, context_info in dataset.items():
        context_text = context_info["context"]
        assert type(context_text) == str
        assert type(context_id) == str
        assert context_id not in context_id_set
        context_id_set.add(context_id)
        for question_answer in context_info["qas"]:
            question_id = question_answer["query_id"]
            question_text = question_answer["question"]
            assert type(question_id) == str
            assert type(question_text) == str
            assert question_id not in question_id_set
            question_id_set.add(question_id)
            question_text = question_text.strip()
            assert len(question_text) > 0
            assert "answer" in question_answer
            check_answer(question_answer["answer"])
            if "validated_answers" in question_answer:
                for answer in question_answer["validated_answers"]:
                    check_answer(answer)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    validate_dataset(os.path.join(dir_path, "../fixtures/vimarc.json"))
