from win32com.client import Dispatch

file_paths = [
    r"D:\thesis\21land\merged_land_sorted.xlsx"
]

xl = Dispatch('Excel.Application')

for file_path in file_paths:
    wb = xl.Workbooks.Open(file_path)
    wb.SaveAs(file_path[:-1], FileFormat=56)
    wb.Close()

xl.Quit()
