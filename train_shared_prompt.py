"""Single additive prompt + prediction head; no retrieval and no key loss."""
from train_head_only import main


if __name__ == '__main__':
    main(baseline='shared_prompt')
