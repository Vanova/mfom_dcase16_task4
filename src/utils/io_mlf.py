"""
Input/output api for MLF label format (HTK label format)
- MLF to DCASE
- DCASE to MLF
"""
import os.path as path
import pandas as pd
import numpy as np
import pprint
import src.utils.io as util_io


def mlf_dcase(mlf_ali):
    search_dir = '/home/vano/wrkdir/datasets/OGI-ML'
    wav_paths = util_io.search_files(search_dir)
    res = pd.DataFrame()
    for fid, events in mlf_ali.items():
        audio_file = filter(lambda x: fid in x, wav_paths)
        if len(audio_file) > 1:
            raise Exception('Found files: %s' % audio_file)
        elif len(audio_file) == 0:
            raise Exception('File is not found!!!')

        audio_file = audio_file[0].replace(search_dir, '')
        starts = np.array(events)[:, 0].astype(int) / 10. ** 7
        ends = np.array(events)[:, 1].astype(int) / 10. ** 7
        labs = np.array(events)[:, 2]
        df_ali = pd.DataFrame(zip([audio_file] * len(starts), starts, ends, labs),
                              columns=['file', 'start', 'end', 'class_label'])
        res = res.append(df_ali)
    return res


if __name__ == '__main__':
    # example of interaction
    ogi_meta_dir = '/home/vano/wrkdir/projects/Python_Projects/attribute_detection/data/ogits_meta/'
    ATTRIBUTES = ['manner', 'place', 'fusion']
    LANGUAGES = ['english', 'german', 'hindi', 'japanese', 'mandarin', 'spanish']

    # loading phonemes to attributes mapping
    for at in ATTRIBUTES:
        for lang in LANGUAGES:
            map_file = path.join(ogi_meta_dir, 'mapping', at, '%s_ph_at.txt' % lang)
            ph_at = util_io.load_dictionary(map_file)
            print((at, lang))
            pprint.pprint(ph_at)
            print

    for lang in LANGUAGES:
        ali_file = path.join(ogi_meta_dir, 'phonemes', '%s.mlf' % lang)
        mlf_ali = util_io.load_mlf(ali_file, 'lab')
        print('Number of %s files: %d' % (lang, len(mlf_ali.keys())))

    # def merge_two_dicts(x, y):
    #     from itertools import chain
    #     from collections import defaultdict
    #     z = defaultdict(list)
    #     for k, v in chain(x.items(), y.items()):
    #         z[k].extend(v)
    #         z[k] = list(set(z[k]))
    #         z[k].sort()
    #     return z
    #
    #
    # for lang in LANGUAGES:
    #     map_file1 = path.join(ogi_meta_dir, 'mapping', ATTRIBUTES[0], '%s_ph_at.txt' % lang)
    #     map_file2 = path.join(ogi_meta_dir, 'mapping', ATTRIBUTES[1], '%s_ph_at.txt' % lang)
    #     ph_at1 = util_io.load_dictionary(map_file1)
    #     ph_at2 = util_io.load_dictionary(map_file2)
    #     fuse = merge_two_dicts(ph_at1, ph_at2)
    #
    #     fuse_file = path.join(ogi_meta_dir, 'mapping', 'fusion', '%s_ph_at.txt' % lang)
    #     with open(fuse_file, 'w') as f:
    #         phones = fuse.keys()
    #         phones.sort()
    #         for k in phones:
    #             line = ' '.join(fuse[k])
    #             f.write(k + ' ' + line + '\n')

    # map phonemes to attributes
    for att in ATTRIBUTES:
        att_alignments = {}
        cnt = 0
        for lang in LANGUAGES:
            mfile = path.join(ogi_meta_dir, 'mapping', att, '%s_ph_at.txt' % lang)
            ph_att = util_io.load_dictionary(mfile)

            afile = path.join(ogi_meta_dir, 'phonemes', '%s.mlf' % lang)
            alignment = util_io.load_mlf(afile)

            for file_id, events in alignment.items():
                for ev in events:
                    labels = ph_att[ev[-1]]
                    ev[-1] = ';'.join(labels)
            cnt += len(alignment)
            print('# of alignments %s' % len(alignment))
            att_alignments.update(alignment)
        print('Number of %s: %d' % (att, cnt))
        print('Number of alignments %s: %d' % (att, len(att_alignments)))
        # save alignments per-attributes
        ali_file = '%s.mlf' % att
        util_io.save_mlf(ali_file, att_alignments)

    # split up dataset
    for dset in ['train', 'test', 'validation']:
        fls = []
        with open(path.join(ogi_meta_dir, 'lists', dset)) as f:
            lines = map(str.strip, f.readlines())
            fls.extend(lines)
            fls.sort()
            fls = map(lambda x: x.replace('.wav', ''), fls)

        for att in ATTRIBUTES:
            afile = path.join('%s.mlf' % att)
            alignment = util_io.load_mlf(afile)
            sub_ali = dict((f, alignment[f]) for f in fls)
            util_io.save_mlf('%s_%s.mlf' % (att, dset), sub_ali)

    # map mlf to dcase format
    # import os
    # import fnmatch
    # wav_paths = []
    # search_dir = '/home/vano/wrkdir/datasets/OGI-ML'
    # for root, dirnames, filenames in os.walk(search_dir):
    #     # filter(lambda x: fid in x, wav_paths)
    #     for filename in fnmatch.filter(filenames, '*.wav'):
    #         wav_paths.append(os.path.join(root, filename))


    # map mlf to dcase format
    for att in ATTRIBUTES:
        afile = '%s.mlf' % att
        alignment = util_io.load_mlf(afile)
        res = mlf_dcase(alignment)
        res.reset_index(drop=True, inplace=True)
        res.to_csv('%s.csv' % att, index=False, header=False, sep='\t')

