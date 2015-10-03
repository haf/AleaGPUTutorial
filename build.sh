# replace external address with internal one.
sed -i.bak 's/https.*8543/http:\/\/192.168.10.77:8111/g' paket.lock


$mono64 ".paket/paket.bootstrapper.exe"
$mono64 ".paket/paket.exe" restore
$mono64 "packages/FAKE/tools/FAKE.exe" buildUnix.fsx Tests $1
