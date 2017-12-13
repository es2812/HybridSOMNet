entrada = dlmread('TEST.PESOS.txt',' ')
lines = dlmread('digitos.entrena.normalizados.txt',' ')
numero_instancias = length(entrada(:,1))
tamanyo_salida = 10

fid = fopen('INPUTMLP.txt','w')

i=1
while i < numero_instancias
  dlmwrite(fid,entrada(i,1:length(entrada)-1)," ","-append")
  dlmwrite(fid,lines(i*2,1:tamanyo_salida)," ","-append")
  i++
end