# Assistente de IA local com tecnologias open source

O projeto usa Ollama para inferência local, Qwen 2.5 3B (Apache-2.0) como
modelo de conversa, Matrix/Synapse para mensagens, PostgreSQL com pgvector
para recuperação de documentos e Arize Phoenix para observabilidade. O
WhatsApp pode ser conectado opcionalmente pela API proprietária da Twilio;
essa integração não faz parte de uma stack exclusivamente open source.
Instagram, Facebook e LinkedIn não estão integrados.

## Pré-requisitos

- Docker Desktop com espaço livre suficiente para as imagens e os modelos.
- Os modelos serão baixados automaticamente no primeiro `docker compose up`.

Inicie os serviços:

```bash
docker compose up --build -d
```

O Ollama roda em um container Docker e os modelos `qwen2.5:3b` e
`nomic-embed-text` (Apache-2.0) são armazenados no volume persistente
`ollama_data`. O primeiro início baixa os modelos e pode demorar. No Mac,
Ollama no Docker usa CPU por padrão; reserve memória suficiente no Docker
Desktop para executar o modelo.

## Usar a API local

Abra a interface de chat em <http://localhost:8501>. Ela exibe uma saudação,
permite conversar com o modelo e começar uma nova conversa.

Para testar a API diretamente:

```bash
curl -X POST http://localhost:8050/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Olá!"}'
```

`GET /health` verifica se a API iniciou. `/ws/voice` recebe mensagens JSON com
um campo `text` e devolve respostas em texto; transcrição e síntese de voz
continuam sendo responsabilidade do cliente.

## Ativar o bot Matrix

O Synapse local fica disponível apenas em `localhost:8008`. Crie uma conta do
bot no servidor:

```bash
docker compose exec synapse register_new_matrix_user \
  -c /data/homeserver.yaml -u assistente --admin http://localhost:8008
```

O comando solicita a senha. Obtenha um token de acesso Matrix com a API de
login (substitua a senha):

```bash
curl -s http://localhost:8008/_matrix/client/v3/login \
  -H 'Content-Type: application/json' \
  -d '{"type":"m.login.password","user":"assistente","password":"SUA_SENHA"}'
```

Crie um arquivo `.env` na raiz com os valores retornados/configurados:

```dotenv
MATRIX_USER_ID=@assistente:matrix.local
MATRIX_ACCESS_TOKEN=seu_token_de_acesso
```

Recrie a aplicação para iniciar o bot:

```bash
docker compose up -d --force-recreate app
```

Convide `@assistente:matrix.local` para uma sala. O bot responde a mensagens
de texto. Mantenha o token privado e não o envie ao controle de versão.

## Ativar WhatsApp via Twilio (opcional)

Esta integração exige uma conta Twilio e um remetente WhatsApp habilitado.
Para receber mensagens no bot e responder automaticamente, também é necessário
um endpoint HTTPS público para webhooks. O app atualmente publica a porta 8050
apenas em `localhost`; configure um reverse proxy HTTPS público para encaminhar
`/webhook/whatsapp` para essa porta. Defina `TWILIO_WEBHOOK_URL` com a URL HTTPS
externa exata, incluindo `/webhook/whatsapp`, e configure essa URL como webhook
de mensagens no console da Twilio.

Gere credenciais novas no console da Twilio. Se uma SID de API e seu segredo
foram compartilhados em chat ou em outro canal não privado, revogue essa chave
e crie outra antes de usá-la. Não coloque credenciais no código nem as envie
em mensagens. Adicione ao arquivo `.env` local, que é ignorado pelo Git:

```dotenv
TWILIO_ACCOUNT_SID=AC...
TWILIO_API_KEY_SID=SK...
TWILIO_API_KEY_SECRET=seu_segredo_da_api_key
TWILIO_AUTH_TOKEN=seu_auth_token_da_conta
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
TWILIO_WEBHOOK_URL=https://seu-dominio.example/webhook/whatsapp
```

`TWILIO_API_KEY_SID` e `TWILIO_API_KEY_SECRET` autenticam o envio de mensagens.
`TWILIO_AUTH_TOKEN` é a credencial da conta usada para validar assinaturas de
webhooks; não é intercambiável com o segredo da API Key. Configure o remetente
conforme o número/sandbox WhatsApp provisionado na Twilio. Depois de salvar o
`.env`, reconstrua o app:

```bash
docker compose up --build -d app
```

### Enviar do Streamlit para o WhatsApp

Para envio manual pela interface, basta configurar no `.env`
`TWILIO_ACCOUNT_SID`, `TWILIO_API_KEY_SID`, `TWILIO_API_KEY_SECRET` e
`TWILIO_WHATSAPP_FROM`. `TWILIO_AUTH_TOKEN` e `TWILIO_WEBHOOK_URL` são
necessários para receber mensagens e validar webhooks, mas não para o envio
manual. Reinicie o app após configurar:

```bash
docker compose up --build -d app ui
```

Abra <http://localhost:8501>, preencha o número de destino no formato
internacional (por exemplo, `+5511999999999`) e o texto, e clique em
**Enviar pelo WhatsApp**. O destinatário precisa estar habilitado para o
remetente configurado (por exemplo, ter entrado no Sandbox). A mensagem pode
ter até 1.600 caracteres. Fora da janela de atendimento do WhatsApp, a Twilio
pode exigir o uso de um template aprovado.

## Observabilidade

Phoenix fica disponível em <http://localhost:6007> (porta interna 6006).
O SDK oficial `phoenix.otel.register()` configura o provedor de traces para o
projeto `llmops-assistant` e instrumenta automaticamente o LangChain. Cada
chamada gera um trace com o span da aplicação e spans filhos do
LangChain/LangGraph. Os spans incluem canal, modelo, intenção, duração,
quantidade de documentos RAG, tamanhos de entrada/saída e tokens quando o
modelo os informa. O Phoenix agrupa chamadas da mesma conversa como sessão.
O projeto aparece no Phoenix após a primeira chamada rastreada ser recebida.

Na API HTTP, reutilize o `session_id` devolvido na resposta enviando-o no
cabeçalho `X-Session-ID` da próxima chamada. Matrix e WhatsApp recebem IDs de
sessão estáveis por sala/remetente; o WebSocket mantém a sessão durante a
conexão. As entradas e respostas são gravadas nos spans para exibição da
conversa na tela Sessions do Phoenix; não envie dados sensíveis se não quiser
que sejam armazenados localmente pelo Phoenix.

O Phoenix é focado em traces/spans e apresenta atributos e análises derivados
deles; esta configuração não exporta métricas OpenTelemetry independentes.
Os atributos numéricos `app.duration_ms`, `app.rag.documents`,
`app.input.length`, `app.output.length` e `llm.token_count.*` podem ser
inspecionados e agregados nos traces/sessões do projeto `llmops-assistant`.
Os dados do PostgreSQL e do Synapse são persistidos em volumes Docker.

Se a autenticação do Phoenix estiver habilitada, crie um System API Key nas
configurações do Phoenix e defina `PHOENIX_API_KEY` no `.env` local. Sem a
chave, o Phoenix pode rejeitar a exportação de traces com HTTP 401.

### Ollama indisponível

Confira o estado e os logs dos containers do Ollama:

```bash
docker compose ps ollama ollama-models
docker compose logs -f ollama ollama-models
```

O serviço `ollama-models` baixa os dois modelos na primeira inicialização. Só
depois que esse serviço terminar com sucesso a API inicia. Se necessário,
repita o download com `docker compose run --rm ollama-models`.
