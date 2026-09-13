Claro. Para testar no Linux, siga esta ordem.

## 1. Obter a branch correta

No servidor Linux, entre no projeto:

```bash
cd /caminho/para/project_multimodal_gen_ai
```

Confirme a branch:

```bash
git branch --show-current
```

Se ainda não tiver a branch local:

```bash
git fetch origin
git switch agents/tmux-train-recovery-orchestrator
git pull
```

Confirme que os arquivos existem:

```bash
test -f train/recovery_orchestrator.py && echo "orchestrator OK"
test -f tests/test_recovery_orchestrator.py && echo "tests OK"
test -f configs/train_queue.csv && echo "queue OK"
```

Você deve estar na raiz do projeto, onde existem:

```text
train/
tests/
configs/
README.md
requirements.txt
```

---

## 2. Ativar o ambiente `venv312`

```bash
source venv312/bin/activate
```

Confirme:

```bash
which python
python --version
test -x venv312/bin/python && echo "venv312 OK"
tmux -V
```

O `which python` deve apontar para algo parecido com:

```text
.../project_multimodal_gen_ai/venv312/bin/python
```

Se o ambiente ainda não existir:

```bash
python3 -m venv venv312
source venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 3. Executar os testes automatizados

Execute sempre a partir da raiz do projeto:

```bash
python -m unittest tests.test_recovery_orchestrator -v
```

Ou, de forma alternativa:

```bash
python -m unittest discover \
  -s tests \
  -p 'test_recovery_orchestrator.py' \
  -v
```

O resultado esperado é algo semelhante a:

```text
test_completed_job_is_skipped ... ok
test_pending_job_is_started_and_marked_running ... ok
test_running_job_without_session_is_resumed ... ok

OK
```

Se aparecer:

```text
ModuleNotFoundError: No module named 'tests'
```

você não está na raiz do projeto. Corrija com:

```bash
cd /caminho/para/project_multimodal_gen_ai
source venv312/bin/activate
python -m unittest tests.test_recovery_orchestrator -v
```

---

## 4. Conferir a fila real

Abra o CSV:

```bash
cat configs/train_queue.csv
```

Ele deve conter:

```csv
id,status,command,working_dir,session_name,resume_override
encoder-seq-v2,pending,venv312/bin/python -m train.train_encoder_seq --config-name chess_encoder_seq_v2,.,train-encoder-seq-v2,checkpoint_load_path=auto
decoder-v12,pending,venv312/bin/python -m train.train_chessdecoder --config-name chess_decoder_player_v12,.,train-decoder-v12,checkpoint_load_path=auto
```

Para uma fila nova:

- use `pending` para treinos que ainda não começaram;
- use `running` somente se existe checkpoint e o treino deve ser retomado;
- use `completed` para treinos já concluídos.

Antes do teste real, confira o Python usado na fila:

```bash
test -x venv312/bin/python && echo "Python da fila OK"
```

---


## 5. Testar se o linger esta ativado (necessario para poder logoff)
To check the current linger status for your user:

```bash
loginctl show-user "$USER" -p Linger
```

Possible outputs:

```text
Linger=yes
```

or:

```text
Linger=no
```

To enable linger:

```bash
loginctl enable-linger "$USER"
```

If you get a permissions error, use:

```bash
sudo loginctl enable-linger "$USER"
```

Then verify again:

```bash
loginctl show-user "$USER" -p Linger
```

Expected:

```text
Linger=yes
```
## 6. Testar uma execução manual

Ainda com o ambiente ativado:

```bash
python -m train.recovery_orchestrator \
  --csv configs/train_queue.csv
```

Como o primeiro treino está `pending`, o orquestrador deverá:

1. Criar a sessão `train-encoder-seq-v2`.
2. Alterar o status para `running`.
3. Executar o treino dentro do `tmux`.

Confira:

```bash
cat configs/train_queue.csv
tmux ls
```

O resultado deverá incluir:

```text
train-encoder-seq-v2: 1 windows
```

Para acompanhar o treinamento:

```bash
tmux attach -t train-encoder-seq-v2
```

Para sair sem interromper:

```text
Ctrl+B
D
```

Ou veja a saída sem anexar:

```bash
tmux capture-pane \
  -p \
  -t train-encoder-seq-v2:0.0 \
  -S -100
```

---

## 7. Testar a recuperação após falha

Se houver uma sessão antiga com o caminho incorreto, encerre-a:

```bash
tmux kill-session -t train-encoder-seq-v2
```

Marque o treino como `running`:

```bash
sed -i 's/^encoder-seq-v2,pending,/encoder-seq-v2,running,/' configs/train_queue.csv
```

Confirme:

```bash
grep '^encoder-seq-v2' configs/train_queue.csv
```

Execute novamente:

```bash
python -m train.recovery_orchestrator \
  --csv configs/train_queue.csv
```

O comando será iniciado com:

```text
checkpoint_load_path=auto
```

Acompanhe:

```bash
tmux ls
tmux capture-pane -p -t train-encoder-seq-v2:0.0 -S -100
```

Importante: o checkpoint mais recente precisa estar no local esperado pela configuração `chess_encoder_seq_v2`.

---

## 8. Testar o `systemd`

Primeiro confirme os caminhos do serviço:

```bash
grep -E 'WorkingDirectory|ExecStart' deploy/train-recovery.service
```

Atualmente ele espera o projeto em:

```text
~/project_multimodal_gen_ai
```

Se o projeto estiver em outro local, edite:

```bash
nano deploy/train-recovery.service
```

Por exemplo:

```ini
[Service]
Type=oneshot
WorkingDirectory=/users/eleves-a/2023/ariel.silva-claudino/project_multimodal_gen_ai
ExecStart=/users/eleves-a/2023/ariel.silva-claudino/project_multimodal_gen_ai/venv312/bin/python -m train.recovery_orchestrator --csv /users/eleves-a/2023/ariel.silva-claudino/project_multimodal_gen_ai/configs/train_queue.csv
```

Instale as unidades:

```bash
mkdir -p ~/.config/systemd/user

cp deploy/train-recovery.service ~/.config/systemd/user/
cp deploy/train-recovery.timer ~/.config/systemd/user/
```

Recarregue o `systemd`:

```bash
systemctl --user daemon-reload
```

Teste o serviço imediatamente:

```bash
systemctl --user start train-recovery.service
```

Verifique o resultado:

```bash
systemctl --user status train-recovery.service
```

Leia os logs:

```bash
journalctl --user -u train-recovery.service -n 100 --no-pager
```

Se o serviço funcionar, ative o timer:

```bash
systemctl --user enable --now train-recovery.timer
```

Verifique:

```bash
systemctl --user status train-recovery.timer
systemctl --user list-timers
```

O timer deverá executar o serviço a cada 30 segundos.

---

## 9. Garantir execução depois do logout

Para manter o serviço do usuário ativo após sair do SSH:

```bash
loginctl enable-linger "$USER"
```

Confira:

```bash
loginctl user-status "$USER"
```

---

## 10. Comandos de acompanhamento

Estado da fila:

```bash
cat configs/train_queue.csv
```

Sessões `tmux`:

```bash
tmux ls
```

Saída do treino:

```bash
tmux capture-pane -p \
  -t train-encoder-seq-v2:0.0 \
  -S -100
```

Logs do orquestrador:

```bash
journalctl --user -u train-recovery.service -f
```

Estado do timer:

```bash
systemctl --user status train-recovery.timer
```

Executar uma verificação imediatamente:

```bash
systemctl --user start train-recovery.service
```

---

## Fluxo completo recomendado

```bash
cd /caminho/para/project_multimodal_gen_ai

git switch agents/tmux-train-recovery-orchestrator
git pull

source venv312/bin/activate

python -m unittest tests.test_recovery_orchestrator -v

python -m train.recovery_orchestrator \
  --csv configs/train_queue.csv

tmux ls
cat configs/train_queue.csv
```

Depois que a execução manual funcionar:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/train-recovery.service ~/.config/systemd/user/
cp deploy/train-recovery.timer ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user start train-recovery.service
systemctl --user enable --now train-recovery.timer

systemctl --user status train-recovery.timer
journalctl --user -u train-recovery.service -f
```

O estado esperado enquanto o primeiro treino estiver ativo é:

```csv
encoder-seq-v2,running,...
decoder-v12,pending,...
```

Quando o primeiro terminar normalmente:

```csv
encoder-seq-v2,completed,...
decoder-v12,running,...
```

Se ele cair, permanecerá `running` e será relançado na próxima execução do timer com `checkpoint_load_path=auto`.

