import sys


def chunk(tags):
    chunks = []
    i = 0
    while i < len(tags):

        start = i
        end = start + 1

        if tags[i].startswith('B') and i+1 < len(tags):

            tag = tags[start][-3:]

            while tags[end] == "I-" + tag:
                end += 1

            if tags[end] == "E-" + tag:
                chunks.append((tag, start, end))
                i = end

        elif tags[i].startswith('S'):

            tag = tags[i][-3:]
            chunks.append((tag, i, i))

        i += 1

    return set(chunks)

if __name__ == '__main__':
    with open(sys.argv[1]) as gold:
        target = chunk([tag.strip() for tag in gold])

    with open(sys.argv[2]) as guesses:
        predicted = chunk([tag.strip() for tag in guesses])

    for name_type in ['GEO', 'ORG', 'OTH', 'PRS']:
        truth = {t for t in target if t[0] == name_type}
        guess = {g for g in predicted if g[0] == name_type}

        predict = len(guess)
        total = len(truth)
        correct = len(truth & guess)

        precision = correct / predict
        recall = correct / total
        f1 = 2 * precision * recall / (precision + recall)

        print(f"""{name_type}
        Precision: {precision}
        Recall: {recall}
        F1: {f1}
        """)
