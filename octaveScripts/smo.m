lines = dlmread('../originalFiles/digitos.test.normalizados.txt',' ')
longitud_fichero = length(lines(:,1))
tamanyo_entrada = length(lines(1,:))

#el fichero esta formateado como [entrada,empty line,salida,empty line] por lo que nos interesa
#cada cuarta linea de la matriz

indices_entradas = [1:4:longitud_fichero]
numero_instancias = length(indices_entradas)
entrada = lines(indices_entradas,:)

#se añade una dimension a cada instancia para evitar igualdad de vectores al normalizar
entrada = [entrada,ones(numero_instancias,1)]

#se calcula la norma de cada instancia como la raiz cuadrada de la suma de los cuadrados
#y se divide cada componente del vector por esta norma, tal quedan vectores de norma 1.
normas_entradas = sqrt(sum(entrada.^2,2))
entrada = entrada./normas_entradas 

#SMO. CONSTANTES:
n_iteraciones = 20
filas_smo = 12
columnas_smo = 8
radio_vecindad = ceil(min(filas_smo,columnas_smo)/2)-1 
#incialmente cubre el maximo de la dimension mas pequeña sin overlap

n_neuronas = filas_smo*columnas_smo #tamaño del espacio de salida
tam_espacio_entrada = tamanyo_entrada+1#por la coordenada extra

#inicializamos pesos aleatorios (entre -5 y 5)
#p[i][j] = peso de la neurona i para la dimension del espacio de entrada j
pesos = rand(n_neuronas,tamanyo_entrada)
#normalizamos
pesos = [pesos,ones(n_neuronas,1)] #añadida coordenada extra
normas_pesos = sqrt(sum(pesos.^2,2))
pesos = pesos./normas_pesos

#todos los vectores entrada y pesos son unitarios, por lo que podemos utilizar
#el coseno del angulo entre ellos. El maximo coseno implicara la neurona mas cercana:
#
#   analiticamente: x w = producto escalar = sum(x_i*w_i)
#   geometricamente: x w = |x||w|cos(x,w) = 1*1*cos(x,w) = cos(x,w)
#
#por lo que podemos calcular el coseno del angulo como sum(x_i*w_i)

#necesitamos los cosenos solo durante la muestra actual, nos sirve un vector
cosenos = zeros(1,n_neuronas)
i = 1
#por cada muestra
while i<=numero_instancias #TODO: ufuncs
  cosenos = sum(entrada(i,:).*pesos,2) 
  [x,ganadora] = max(cosenos)#x se desecha, es el valor, nos interesa el indice
  
  #calculamos la fila y columna de la ganadora para hacer mas facil calculos siguientes
  #indice = (fila-1)*NUMCOL + columna
  #fila = floor(indice/NUMCOL)+1 (se suma 1 ya que se indexa por 1)
  #columna = mod(indice,NUMCOL)+1 (idem)
  #tras esto se las multiplica por el numero de filas/columnas respectivamente para asegurar aciclidad
  fila_ganadora = ceil(ganadora/columnas_smo)
  columna_ganadora = mod(ganadora,columnas_smo)
  
  if columna_ganadora == 0
    columna_ganadora = columnas_smo
  endif
  
  #modificamos las vecinas
  
  #se añade una fila a recorrer por cada unidad que aumenta el radio, mas la fila de la ganadora
  filas_a_recorrer = (radio_vecindad*2)+1
  #idem
  columnas_a_recorrer = (radio_vecindad*2)+1
  
  recorriendo_fila = fila_ganadora-radio_vecindad
  
  indices_vecinas = zeros(n_neuronas,1)
  indices_vecinas(1) = ganadora
  
  iz = 2
  fi=1  
  while fi<=filas_a_recorrer
    recorriendo_columna = columna_ganadora-radio_vecindad 
    co=1
    while co<=columnas_a_recorrer
    
      fila_vecina = mod(recorriendo_fila,filas_smo)
      if fila_vecina == 0
        fila_vecina = filas_smo
      endif
      
      columna_vecina = mod(recorriendo_columna,columnas_smo)
      if columna_vecina == 0
        columna_vecina = columnas_smo
      endif
      
      #necesario calcular el indice?
      indice = (fila_vecina-1)*columnas_smo + columna_vecina
      
      if indice==ganadora
        #no modificar indice
      endif
      #modificar indice
      indices_vecinas(iz) = indice
      recorriendo_columna++
      co++
      iz++
    end
    recorriendo_fila++
    fi++
  end
  i++
end
#alfa = alfa_0/(1+(t/n_muestras)) t AUMENTA POR CADA MUESTRA, PERO NO VUELVE A 0 TRAS FIN DE EPOCA
#alfa inicial no entre 0 y 1 a ojo (10, 20)
#etiquetado por neuronas (necesitas las salidas)
#luego se le da a un mlp (codigo en carpeta TAA de onedrive)

