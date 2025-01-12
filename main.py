import pengenalancuaca
import os
lanjut = True
while (lanjut):
    model,kelas = pengenalancuaca.build_model()
    print("Selamat datang pada program pengenalan cuaca")
    path = input("Silahkan input path dari gambar yang ingin diprediksi: ")
    hasil_prediksi = pengenalancuaca.klasifikasi_cuaca(path,model,kelas)
    print(f"Gambar tersebut diprediksi memiliki cuaca {hasil_prediksi}")
    tanya = input("Apakah ingin memprediksi gambar lain (y/n): ")
    if(tanya == "y"):
        lanjut = True
    else:
        lanjut = False
    os.system('cls')