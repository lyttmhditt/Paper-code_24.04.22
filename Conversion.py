# -*- coding: utf-8 -*-
import csv
import os
import xlrd
import xlwt


def csv_2_xls():
    csvfiles = os.listdir('.')
    csvfiles = filter(lambda x: x.endswith('csv'), csvfiles)
    for csvfile in list(csvfiles):
        finename = csvfile.split('.')[0]
        if not os.path.exists('excel'):
            os.mkdir('excel')
        xlsfile = 'excel/' + finename + '.xls'
        with open(csvfile, 'r') as f:
            reader = csv.reader(f)
            workbook = xlwt.Workbook()
            sheet = workbook.add_sheet('sheet1')  # 创建一个sheet表格
            i = 0
            for line in reader:
                j = 0
                for v in line:
                    sheet.write(i, j, v)
                    j += 1
                i += 1
            workbook.save(xlsfile)  # 保存Excel
        print(f'转换完成: {csvfile} -> {xlsfile}')


def xls_2_csv():
    xlsfiles = os.listdir('.')
    xlsfiles = filter(lambda x: x.endswith('xls'), xlsfiles)
    for xlsfile in list(xlsfiles):
        book = xlrd.open_workbook(xlsfile)
        table = book.sheets()[0]
        nrows = table.nrows
        finename = xlsfile.split('.')[0]
        if not os.path.exists('csv'):
            os.mkdir('csv')
        csvfile = 'csv/' + finename + '.csv'
        with open(csvfile, 'w') as f:
            writer = csv.writer(f)
            for i in range(nrows):
                writer.writerow(table.row_values(i))
        print(f'转换完成: {xlsfile} -> {csvfile}')


if __name__ == '__main__':
    csv_2_xls()
    xls_2_csv()
