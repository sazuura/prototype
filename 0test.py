import MetaTrader5 as mt5

#Koneksi
if not mt5.initialize():
    print("Gagal koneksi ke MetaTrader 5")
    quit()

#Cek Pair yang tersedia
symbols = mt5.symbols_get()
for s in symbols:
    print(s.name)

#Cek Akun
account = mt5.account_info()
print("Account: ", account)

#Ambil data pair terbaru 
rates = mt5.copy_rates_from_pos("EURUSDc", mt5.TIMEFRAME_H1, 0, 10)
print("Last candle: ")
for x in rates:
       print(x)

mt5.shutdown()