import os
import numpy as np
import yaml
import configparser


class Load():
    def __init__(self, filename, filepath, NETA_BIS = True) :
        self.NETA_BIS = NETA_BIS
        if self.NETA_BIS:
            with open(os.path.join(filepath, filename)) as fp:
                Lines = fp.readlines()
                temp =[]
                for line in Lines:
                    if line.strip().startswith('#'):
                        temp.append(line)
            file1 = "".join(temp[:-1])
            file1 = file1.replace('#', "")
            self.text = yaml.safe_load(file1)
            self.data = np.loadtxt(os.path.join(filepath, filename))
        else:

            config = configparser.ConfigParser()
            config.read('Parameters.ini')
            params_ini = config._sections
        

            with open(os.path.join(filepath, filename)) as fp:
                Lines = fp.readlines()
                temp =[]
                count = 0
                self.text ={}
                self.string =""
                for line in Lines:
                    if not line[0].isdigit():
                        count +=1
                        if not line[0] =='*':
                            temp = (line.strip()).split(':')
                            self.text[temp[0].strip()] =  temp[1]
                            self.string += line

            self.data = np.loadtxt(os.path.join(filepath, filename), skiprows =count, delimiter=';')

            if self.text['DeltaF'].strip().isdigit():
                DeltaF = int(self.text['DeltaF'])
            else:
                DeltaF =int(params_ini['laser_params']['deltaf'])

            fs = int(self.text['SampleRate'])
            Freq_laser = 42e6
            dt = (DeltaF/Freq_laser/fs)*1e9
            timeaxis =np.arange(len(self.data))*dt
            self.data[:,0] = timeaxis


    def pretty_print(self):
        
        if self.NETA_BIS:
            return yaml.dump(self.text, default_flow_style=False)
        else:
            return self.string




if __name__ == '__main__':

    filepath = r"C:\Users\sandeep\Desktop\Inbox\I2T2M\SAFRAN_fiber\Data_LAUM\2022\ToMathieu\neta_bis"
    filename = "IM712k91M002473_top_btm_20221127_173523.txt"

    filepath = r"C:\Users\sandeep\Desktop\Inbox\I2T2M\SAFRAN_fiber\Data_LAUM\2022\ToMathieu\neta"
    filename ="IM712KGP_910M0042164_10e6-pd-Gain_16112022_15_53_20.txt"
    da =Load(filename, filepath,False)
    print(da.pretty_print())