def lvid(runw, runk):
    W = ['daria', 'denis', 'eli', 'ido', 'ira', 'lena', 'lyova', 'moshe', 'shahar']
    K = range(1, 26, 1)
    W_act = ['bend', 'side', 'skip', 'wave1']
    K_act = ['jogging', 'walking']
    K_dir = range(1, 5, 1)
    endings = ['.avi', '_uncomp.avi', '_train.txt']
    folder = 'c:/Python27/Scripts/videos/'
    if runw == 1:
        for w in W:
            for i, wa in enumerate(W_act):
                fname = w + '_' + wa
                f = open(folder + fname + endings[2], 'w')
                f.write('"' + fname + endings[0] + '"')
                if i == 0:
                    f.write(' (0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0)')
                elif i == 1:
                    f.write(' (0, 1, 0, 2, 2, 1, 1, 3, 0, 0, 3, 3)')
                elif i == 2:
                    f.write(' (1, 0, 0, 1, 2, 1, 1, 4, 0, 0, 4, 1)')
                else:
                    f.write(' (0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0)')
                f.close()

    if runk == 1:
        for k in K:
            for j, ka in enumerate(K_act):
                for d in K_dir:
                    if k < 10:
                        z = '0'
                    else:
                        z = ''
                    fname = 'person' + z + str(k) + '_' + ka + '_d' + str(d)
                    f = open(folder + fname + endings[2], 'w')
                    f.write('"' + fname + endings[1] + '"')
                    if j == 0:
                        f.write(' (2, 2, 0, 1, 0, 1, 1, 2, 2, 2, 2, 2)')
                    else:
                        f.write(' (2, 2, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1)')
                    f.close()
                
