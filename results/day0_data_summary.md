# Day 0 data summary

Generated 2026-10-21 20:30; pandas 3.0.6, pyarrow 26.0.0

## IBM AML HI-Small_Trans.csv

- file size: 475.7 MB
- note: the raw header has two columns both named `Account`; renamed here to From Account / To Account
- load time: 0.3s
- rows: **5,078,345**, columns: 11
- column names: ['Timestamp', 'From Bank', 'From Account', 'To Bank', 'To Account', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering']

Dtypes:
```
Timestamp             datetime64[us]
From Bank                        str
From Account                     str
To Bank                          str
To Account                       str
Amount Received              float64
Receiving Currency               str
Amount Paid                  float64
Payment Currency                 str
Payment Format                   str
Is Laundering                  int64
```
First 3 rows:
```
            Timestamp From Bank From Account To Bank To Account  Amount Received Receiving Currency  Amount Paid Payment Currency Payment Format  Is Laundering
0 2022-09-01 00:20:00       010    8000EBD30     010  8000EBD30          3697.34          US Dollar      3697.34        US Dollar   Reinvestment              0
1 2022-09-01 00:20:00     03208    8000F4580     001  8000F5340             0.01          US Dollar         0.01        US Dollar         Cheque              0
2 2022-09-01 00:00:00     03209    8000F4670   03209  8000F4670         14675.57          US Dollar     14675.57        US Dollar   Reinvestment              0
```
Null counts: none
- time range: 2022-09-01 00:00:00 -> 2022-09-18 16:18:00
- Is Laundering positives: **5,177** (0.1019%); negatives 5,073,168
- unique accounts (bank+account): **515,088**; unique account numbers only: 515,080
- payment formats (count / laundering rate):
```
                   size      mean
Payment Format                   
Cheque          1864331  0.000174
Credit Card     1323324  0.000156
ACH              600797  0.007462
Cash             490891  0.000220
Reinvestment     481056  0.000000
Wire             171855  0.000000
Bitcoin          146091  0.000383
```
- duplicate rows (all columns): 9

## IBM AML HI-Small_accounts.csv

- file size: 34.1 MB
- load time: 0.4s
- rows: **518,581**, columns: 5
- column names: ['Bank Name', 'Bank ID', 'Account Number', 'Entity ID', 'Entity Name']

Dtypes:
```
Bank Name         str
Bank ID           str
Account Number    str
Entity ID         str
Entity Name       str
```
First 3 rows:
```
             Bank Name Bank ID Account Number  Entity ID                 Entity Name
0  Portugal Bank #4507  331579      80B779D80  80062E240  Sole Proprietorship #50438
1      Canada Bank #27     210      809D86900  800C998A0          Corporation #33520
2          UK Bank #33   21884      80812BE00  800C47F50          Partnership #35397
```
Null counts: none
- unique Account Number: 518,573; unique Entity ID: 166,207; unique Bank ID: 30,470
- note: trans zero-pads bank ids (e.g. `010`), accounts does not (`10`); bank ids are compared as integers
- **Join**: accounts.(Bank ID, Account Number) <-> trans.(From Bank, From Account) and (To Bank, To Account); Entity ID groups accounts by owning entity.
- transaction accounts found in accounts table: 515,088 / 515,088
- accounts-table rows never appearing in trans: 3,493
- (Bank ID, Account Number) duplicates in accounts table: 0

## IBM AML HI-Small_Patterns.txt

- file size: 0.3 MB
- total lines: 4,319

First 40 lines:
```
BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
2022/09/01 04:33,021174,800737690,020,80020C5B0,8630.40,Euro,8630.40,Euro,ACH,1
2022/09/01 09:14,021174,800737690,020,80006A5E0,35642.49,Yuan,35642.49,Yuan,ACH,1
2022/09/01 09:56,021174,800737690,00220,8007A5B70,5738987.96,US Dollar,5738987.96,US Dollar,ACH,1
2022/09/01 11:28,021174,800737690,001244,80093C0D0,7254.53,US Dollar,7254.53,US Dollar,ACH,1
2022/09/01 13:13,021174,800737690,00513,80078E200,6990.87,US Dollar,6990.87,US Dollar,ACH,1
2022/09/01 14:11,021174,800737690,020,80066B990,12536.92,Euro,12536.92,Euro,ACH,1
2022/09/02 15:40,021174,800737690,00410,8002CC310,3511.82,Euro,3511.82,Euro,ACH,1
2022/09/02 21:23,021174,800737690,01292,8004030A0,16135.09,US Dollar,16135.09,US Dollar,ACH,1
2022/09/02 23:10,021174,800737690,01601,800578800,12183.28,US Dollar,12183.28,US Dollar,ACH,1
2022/09/03 09:29,021174,800737690,001,800AAF0B0,15197.45,US Dollar,15197.45,US Dollar,ACH,1
2022/09/03 14:22,021174,800737690,01420,8006FB970,4717.76,US Dollar,4717.76,US Dollar,ACH,1
2022/09/04 07:35,021174,800737690,010,800449990,19332.59,US Dollar,19332.59,US Dollar,ACH,1
2022/09/04 09:10,021174,800737690,011852,800AEBA90,300.93,Euro,300.93,Euro,ACH,1
2022/09/04 12:03,021174,800737690,010,800692550,2724.99,US Dollar,2724.99,US Dollar,ACH,1
2022/09/04 15:48,021174,800737690,011056,8007486D0,15224.53,Euro,15224.53,Euro,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops
2022/09/01 00:03,01467,8013C4030,020,80BC62F10,58702.10,Yuan,58702.10,Yuan,ACH,1
2022/09/01 02:52,020,80BC62F10,0240229,80F025640,7332.87,Swiss Franc,7332.87,Swiss Franc,ACH,1
2022/09/02 08:44,0240229,80F025640,0217,80FD27570,26443.70,Shekel,26443.70,Shekel,ACH,1
2022/09/02 12:29,0217,80FD27570,0024856,8090E8EB0,10621.24,Canadian Dollar,10621.24,Canadian Dollar,ACH,1
2022/09/03 10:20,0024856,8090E8EB0,0071,804ABCE90,637140.60,Rupee,637140.60,Rupee,ACH,1
2022/09/03 12:08,0071,804ABCE90,0213737,805494C30,621578.18,Rupee,621578.18,Rupee,ACH,1
2022/09/03 13:24,0213737,805494C30,014290,801B949C0,7222.58,Euro,7222.58,Euro,ACH,1
2022/09/04 03:24,014290,801B949C0,0010057,803DE1580,892031.21,Yen,892031.21,Yen,ACH,1
2022/09/04 09:44,0010057,803DE1580,0028628,80ACEE280,11364.12,Australian Dollar,11364.12,Australian Dollar,ACH,1
2022/09/04 15:51,0028628,80ACEE280,01467,8013C4030,7945.55,US Dollar,7945.55,US Dollar,ACH,1
END LAUNDERING ATTEMPT - CYCLE

BEGIN LAUNDERING ATTEMPT - GATHER-SCATTER:  Max 3-degree Fan-In
2022/09/01 00:04,0119,811C597B0,0048309,811C599A0,34254.65,Saudi Riyal,34254.65,Saudi Riyal,ACH,1
2022/09/01 19:27,0150240,812D22980,0048309,811C599A0,5971.98,Saudi Riyal,5971.98,Saudi Riyal,ACH,1
2022/09/04 05:06,0222,811B83280,0048309,811C599A0,50445.58,Saudi Riyal,50445.58,Saudi Riyal,ACH,1
2022/09/04 05:03,0048309,811C599A0,0048309,811C599A0,48649.42,Saudi Riyal,48649.42,Saudi Riyal,ACH,1
2022/09/04 14:59,0048309,811C599A0,0119,811C597B0,64379.45,Saudi Riyal,64379.45,Saudi Riyal,ACH,1
END LAUNDERING ATTEMPT - GATHER-SCATTER

```
Attempts per typology:
```
CYCLE               54
GATHER-SCATTER      51
BIPARTITE           49
FAN-OUT             48
SCATTER-GATHER      44
STACK               43
RANDOM              41
FAN-IN              40
TOTAL              370
```
- transaction lines inside attempts: 3,209

**Markup:** each laundering attempt is a block `BEGIN LAUNDERING ATTEMPT - <TYPE>[: detail]` ... `END LAUNDERING ATTEMPT - <TYPE>`.
Between the markers each line is a transaction in exactly the same 11-field CSV layout as `HI-Small_Trans.csv`
(Timestamp, From Bank, From Account, To Bank, To Account, Amount Received, Receiving Currency, Amount Paid,
Payment Currency, Payment Format, Is Laundering). There is no transaction ID, so **linking = join on all of those fields**
(timestamp + from bank/account + to bank/account + amounts + currencies + format); attempt id and typology are then attached
to the matching trans row(s).

- parsed pattern rows: 3,209; matched to a trans row: 3,209 (100.00%); unmatched: 0
- distinct trans rows matched: 3,209; trans positives total: 5,177
- pattern rows matching >1 trans row (ambiguous key): 0
- Is Laundering=1 trans rows NOT in any pattern attempt: 1,968

## Elliptic

- elliptic_txs_features.csv: 689.7 MB
- elliptic_txs_classes.csv: 3.3 MB
- elliptic_txs_edgelist.csv: 4.5 MB

### elliptic_txs_features.csv (no header)

- first line has 167 fields, first value `230425980` is numeric -> confirms no header
- load time: 0.5s
- rows: **203,769**, columns: 167
- column names: ['txId', 'time_step', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6'] ... ['f163', 'f164', 'f165']

Dtypes:
```
float32    165
int64        1
int32        1
```
First 3 rows:
```
        txId  time_step        f1        f2        f3       f4        f5        f6        f7        f8        f9       f10
0  230425980          1 -0.171469 -0.184668 -1.201369 -0.12197 -0.043875 -0.113002 -0.061584 -0.162097 -0.167933 -0.049707
1    5530458          1 -0.171484 -0.184668 -1.201369 -0.12197 -0.043875 -0.113002 -0.061584 -0.162112 -0.167948 -0.049707
2  232022460          1 -0.172107 -0.184668 -1.201369 -0.12197 -0.043875 -0.113002 -0.061584 -0.162749 -0.168576 -0.049707
```
Null counts: none
- node count (rows): **203,769**; unique txId: 203,769
- time step range: 1 - 49 (49 distinct)
- feature columns (excluding txId, time step): 165  (anonymized: 93 local + 72 aggregated neighbour features per the Elliptic paper; no raw amounts, addresses or counterparties)

### elliptic_txs_classes.csv

- rows: 203,769; columns: ['txId', 'class']
```
class
unknown    157205
2           42019
1            4545
```
- txId join features<->classes: {'both': 203769, 'left_only': 0, 'right_only': 0}
- label coding in this file: `1` = illicit, `2` = licit, `unknown` = unlabelled (standard Elliptic coding; the file holds numeric codes, not the words illicit/licit)
Class counts per time step:
```
class        1     2  unknown
time_step                    
1           17  2130     5733
2           18  1099     3427
3           11  1268     5342
4           30  1410     4253
5            8  1874     4921
6            5   480     3843
7          102  1101     4845
8           67  1098     3292
9          248   530     4218
10          18   954     5755
11         131   565     3600
12          16   490     1541
13         291   518     3719
14          43   374     1605
15         147   471     3021
16         128   402     2445
17          99   712     2574
18          52   337     1587
19          80   665     2761
20         260   640     3391
21         100   541     2896
22         158  1605     4131
23          53  1134     2978
24         137   989     3466
25         118   476     1720
26          96   421     2006
27          24   182      883
28          85   199     1369
29         329   845     3101
30          83   441     1959
31         106   604     2106
32         342   981     3202
33          23   418     2710
34          37   478     1971
35         182  1159     4166
36          33  1675     4685
37          40   458     2808
38         111   645     2135
39          81  1102     1577
40         112  1099     3270
41         116  1016     4210
42         239  1915     4986
43          24  1346     3693
44          24  1567     3384
45           5  1216     4377
46           2   710     2807
47          22   824     4275
48          36   435     2483
49          56   420     1978
```

### elliptic_txs_edgelist.csv

- edges: 234,355; columns: ['txId1', 'txId2']
- edge endpoints missing from features: 0
- null counts: {'txId1': 0, 'txId2': 0}

## SAML-D.csv (out of scope for the 5-day plan; load check only)

- file size: 996.2 MB
- load time: 0.9s
- rows: **9,504,852**; columns: ['Time', 'Date', 'Sender_account', 'Receiver_account', 'Amount', 'Payment_currency', 'Received_currency', 'Sender_bank_location', 'Receiver_bank_location', 'Payment_type', 'Is_laundering', 'Laundering_type']
- dtypes (arrow): {'Time': 'time32[s]', 'Date': 'date32[day]', 'Sender_account': 'int64', 'Receiver_account': 'int64', 'Amount': 'double', 'Payment_currency': 'string', 'Received_currency': 'string', 'Sender_bank_location': 'string', 'Receiver_bank_location': 'string', 'Payment_type': 'string', 'Is_laundering': 'int64', 'Laundering_type': 'string'}
