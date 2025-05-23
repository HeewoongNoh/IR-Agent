import glob
import pandas
import nistchempy as nist


search = nist.Search()
fnames = glob.glob('compounds/*.xlsx')
n_succ = 0


for n in range(0, len(fnames)):
    data = pandas.read_excel(fnames[n], header=None).values.tolist()
    fname = fnames[n].split('\\')[1].split('.xlsx')[0]
    metadata = list()

    for compound in data:
        try:
            search.find_compounds(identifier=compound[0], search_type='name')
            search.load_found_compounds()

            if len(search.IDs) > 0:
                c = search.compounds[0]
                c.get_spectra('IR')
                n_succ += 1

                for i in range(0, len(c.IR)):
                    s = c.IR[i]
                    metadata.append([compound[0], search.IDs[0], s.spec_type, int(s.spec_idx)])
                    s.save('{}_{}.jdx'.format(search.IDs[0], i), 'dataset/ir')
                    print('success', n_succ, n, fnames[n], compound[0])
            else:
                metadata.append([compound[0], -1, 'None', -1])
                print('fail', n, fnames[n], compound[0])
        except Exception as e:
            metadata.append([compound[0], -1, 'exception', -1])
            print('Exception\t{}'.format(e))

    pandas.DataFrame(metadata).to_excel('dataset/metadata/metadata_{}.xlsx'.format(fname), index=False, header=False)