# Comprueba que compreface-lite NO hace ninguna conexion de salida (telemetria u otra). Arranca un DNS que registra todas las
# consultas, lanza el servicio apuntando a el, hace peticiones y lista lo que se intento resolver. Debe salir 0 consultas.
# Uso: .\egress_check.ps1 -Image ghcr.io/<usuario>/compreface-lite:1.0.0 -Photo .\una-cara.jpg
param([Parameter(Mandatory)][string]$Image, [Parameter(Mandatory)][string]$Photo, [string]$ModelsVolume = "")
docker container rm -f dnslog egress-test 2>$null | Out-Null
docker network rm egressnet 2>$null | Out-Null
docker network create egressnet | Out-Null
docker run -d --name dnslog --network egressnet alpine sh -c "apk add -q dnsmasq && exec dnsmasq -k --log-queries --log-facility=- --no-resolv --server=1.1.1.1" | Out-Null
Start-Sleep 12
$dnsIp = (docker inspect dnslog | ConvertFrom-Json)[0].NetworkSettings.Networks.egressnet.IPAddress
if (-not $dnsIp) { throw "sin IP de DNS: la prueba no es valida" }
$mount = @(); if ($ModelsVolume) { $mount = @("-v", "${ModelsVolume}:/models:ro") }
docker run -d --name egress-test --network egressnet --dns $dnsIp --read-only --tmpfs "/tmp:size=64m" --cap-drop ALL @mount `
  -e VERIFICATION_API_KEY=k1 -e DETECTION_API_KEY=k2 -p 18099:8080 $Image | Out-Null
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep 2
  try { if ((Invoke-WebRequest http://127.0.0.1:18099/healthcheck -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { break } } catch {}
}
1..6 | ForEach-Object { curl.exe -s -o NUL -X POST "http://127.0.0.1:18099/api/v1/detection/detect?face_plugins=pose,age" -H "x-api-key: k2" -F "file=@$Photo;type=image/jpeg" }
Start-Sleep 75
$q = @(docker logs dnslog 2>&1 | Select-String "query\[" | ForEach-Object { $_.Line })
"consultas de red del servicio: $($q.Count)"
$q | Select-Object -First 10
docker container rm -f dnslog egress-test | Out-Null
docker network rm egressnet | Out-Null
