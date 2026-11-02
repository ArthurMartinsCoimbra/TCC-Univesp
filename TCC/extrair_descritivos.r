
# ============================================================
# 1. CARREGAR PACOTE E BASE DO EXPERIMENTO
# ============================================================

library(tsdl)

base <- read.csv(
  "amostra_tcc.csv",
  stringsAsFactors = FALSE
)

# Verificar se a coluna de identificadores existe
if (!"id_tsdl" %in% names(base)) {
  stop("A coluna id_tsdl não foi encontrada.")
}

if (!"subject" %in% names(base)) {
  stop("A coluna subject não foi encontrada.")
}

# Identificadores únicos das séries utilizadas
ids <- sort(unique(as.integer(base$id_tsdl)))

if (anyNA(ids)) {
  stop("Existem identificadores inválidos na base.")
}

if (any(ids < 1 | ids > length(tsdl))) {
  stop("Algum identificador não corresponde a uma série do TSDL.")
}

if (length(ids) != 16) {
  warning("A base não contém exatamente 16 séries.")
}


# ============================================================
# 2. FUNÇÃO PARA EXTRAIR ATRIBUTOS
# ============================================================

obter_atributo <- function(serie, nome) {

  valor <- attr(serie, nome, exact = TRUE)

  if (is.null(valor) || length(valor) == 0) {
    return(NA_character_)
  }

  paste(as.character(valor), collapse = "; ")
}


# ============================================================
# 3. CONVERTER FREQUÊNCIAS EM PERIODICIDADES
# ============================================================

obter_periodicidade <- function(freq) {

  switch(
    as.character(freq),
    "1" = "Anual",
    "4" = "Trimestral",
    "12" = "Mensal",
    "52" = "Semanal",
    paste("Outra frequência:", freq)
  )
}


# ============================================================
# 4. EXTRAIR INFORMAÇÕES DAS 16 SÉRIES
# ============================================================

lista_descritivos <- lapply(ids, function(id) {

  # Recuperar a série original pelo identificador
  serie <- tsdl[[id]]

  # Recuperar categoria utilizada no TCC
  categoria <- unique(
    base$subject[base$id_tsdl == id]
  )

  categoria <- categoria[!is.na(categoria)]

  if (length(categoria) != 1) {
    stop(
      paste(
        "Categoria inconsistente para a série",
        id
      )
    )
  }

  # Informações temporais
  freq <- frequency(serie)

  inicio <- paste(
    start(serie),
    collapse = "-"
  )

  fim <- paste(
    end(serie),
    collapse = "-"
  )

  data.frame(
    id_tsdl = id,

    descricao = obter_atributo(
      serie,
      "description"
    ),

    fonte = obter_atributo(
      serie,
      "source"
    ),

    categoria_tcc = as.character(categoria),

    categoria_tsdl = obter_atributo(
      serie,
      "subject"
    ),

    frequencia = freq,

    periodicidade = obter_periodicidade(
      freq
    ),

    inicio = inicio,

    fim = fim,

    n_observacoes = length(serie),

    stringsAsFactors = FALSE
  )
})


# ============================================================
# 5. CONSOLIDAR RESULTADOS
# ============================================================

descritivos <- do.call(
  rbind,
  lista_descritivos
)


# ============================================================
# 6. CONFERIR QUANTIDADE DE OBSERVAÇÕES
# ============================================================

quantidade_base <- aggregate(
  indice ~ id_tsdl,
  data = base,
  FUN = length
)

names(quantidade_base)[2] <- "n_base"

descritivos <- merge(
  descritivos,
  quantidade_base,
  by = "id_tsdl",
  sort = TRUE
)

if (any(
  descritivos$n_observacoes != descritivos$n_base
)) {
  warning(
    "A quantidade de observações do TSDL e da amostra diverge."
  )
}

descritivos$n_base <- NULL


# ============================================================
# 7. EXPORTAR CSV
# ============================================================

write.csv(
  descritivos,
  "descritivo_16_series.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)


# ============================================================
# 8. EXPORTAR TXT PARA DOCUMENTAÇÃO
# ============================================================

linhas_txt <- c(
  "DESCRITIVO DAS 16 SÉRIES TEMPORAIS",
  "Fonte: Time Series Data Library (TSDL)",
  ""
)

for (i in seq_len(nrow(descritivos))) {

  item <- descritivos[i, ]

  bloco <- c(
    paste0("SÉRIE ", item$id_tsdl),
    paste0("Descrição: ", item$descricao),
    paste0("Fonte: ", item$fonte),
    paste0("Categoria no TCC: ", item$categoria_tcc),
    paste0("Categoria original TSDL: ", item$categoria_tsdl),
    paste0("Periodicidade: ", item$periodicidade),
    paste0("Frequência: ", item$frequencia),
    paste0("Início: ", item$inicio),
    paste0("Fim: ", item$fim),
    paste0("Observações: ", item$n_observacoes),
    ""
  )

  linhas_txt <- c(
    linhas_txt,
    bloco
  )
}

writeLines(
  linhas_txt,
  "descritivo_16_series.txt",
  useBytes = TRUE
)


# ============================================================
# 9. EXIBIR RESULTADO
# ============================================================

cat("\nDescritivos extraídos com sucesso!\n\n")

print(
  descritivos[
    ,
    c(
      "id_tsdl",
      "descricao",
      "fonte",
      "periodicidade",
      "n_observacoes"
    )
  ],
  row.names = FALSE
)

cat("\nArquivos gerados:\n")
cat("- descritivo_16_series.csv\n")
cat("- descritivo_16_series.txt\n")
