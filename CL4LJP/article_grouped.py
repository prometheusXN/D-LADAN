import json

law_path = '/home/nxu/Ladan_tnnls/data/new_law.txt'
large_law_path = '/home/nxu/Ladan_tnnls/big_data/new_law_big.txt'

Chap2LawDict = {0: list(range(102, 114)),
                1: list(range(114, 140)),
                2: list(range(140, 232)),
                3: list(range(232, 263)),
                4: list(range(263, 277)),
                5: list(range(277, 368)),
                6: list(range(368, 382)),
                7: list(range(382, 397)),
                8: list(range(397, 420)),
                9: list(range(420, 452))}


def neg_dict_generator(Chapter2LawDict, law_label_path):
    index = 0
    law2id_dict = {}
    id2law_dict = {}
    law_list = []
    law2neg_list = {}
    for law in open(law_label_path, 'r', encoding='utf-8').readlines():
        id2law_dict[index] = int(law.strip())
        law2id_dict[int(law.strip())] = index
        law_list.append(int(law.strip()))
        index += 1

    for chapter in Chapter2LawDict.keys():
        law_list_filted = []
        for law_index in Chapter2LawDict[chapter]:
            if law_index in law_list:
                law_list_filted.append(law_index)
            else:
                continue
        Chapter2LawDict[chapter] = law_list_filted

    print(Chapter2LawDict)

    for chapter in Chapter2LawDict.keys():
        if len(Chapter2LawDict[chapter]) == 0:
            continue
        elif len(Chapter2LawDict[chapter]) == 1:
            law2neg_list[Chapter2LawDict[chapter][0]] = []
        else:
            for law in Chapter2LawDict[chapter]:
                candidate_list = Chapter2LawDict[chapter].copy()
                candidate_list.remove(law)
                law2neg_list[law] = candidate_list

    print(law2neg_list)

    idx2neg_list = {}
    for law, neg_list in law2neg_list.items():
        idx2neg_list[law2id_dict[law]] = [law2id_dict[i] for i in neg_list]

    print(idx2neg_list)
    return idx2neg_list


if __name__ == "__main__":

    law_path = '/home/nxu/Ladan_tnnls/data/new_law.txt'
    large_law_path = '/home/nxu/Ladan_tnnls/big_data/new_law_big.txt'

    Chap2LawDict = {0: list(range(102, 114)),
                    1: list(range(114, 140)),
                    2: list(range(140, 232)),
                    3: list(range(232, 263)),
                    4: list(range(263, 277)),
                    5: list(range(277, 368)),
                    6: list(range(368, 382)),
                    7: list(range(382, 397)),
                    8: list(range(397, 420)),
                    9: list(range(420, 452))}

    neg_law_path = '/home/nxu/Ladan_tnnls/CL4LJP/processed_data/CAIL/data/law2neg.json'
    neg_law_big_path = '/home/nxu/Ladan_tnnls/CL4LJP/processed_data/CAIL/big_data/law2neg.json'

    neg_law_dict = neg_dict_generator(Chap2LawDict.copy(), law_path)
    neg_law_big_dict = neg_dict_generator(Chap2LawDict.copy(), large_law_path)
    print(len(neg_law_dict.keys()))
    print(len(neg_law_big_dict.keys()))

json.dump(neg_law_dict, open(neg_law_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
json.dump(neg_law_big_dict, open(neg_law_big_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
