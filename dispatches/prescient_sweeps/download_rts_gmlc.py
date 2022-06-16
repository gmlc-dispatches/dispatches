import os

def download_rts_gmlc( dir_name="DISPATCHES_RTS-GMLC" ):
    cur_path = os.getcwd()
    this_file_path = os.path.dirname(os.path.realpath(__file__))

    if os.path.isdir(os.path.join(this_file_path, dir_name)):
        return
    
    os.chdir(this_file_path)
    os.system(f"git clone --depth=1 https://github.com/bknueven/RTS-GMLC -b no_reserves {dir_name}")
    
    os.chdir(cur_path)

if __name__ == "__main__":
    download_rts_gmlc()
