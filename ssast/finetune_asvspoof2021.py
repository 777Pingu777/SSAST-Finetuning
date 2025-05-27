import os
import argparse
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import subprocess
import tempfile
from src.models.ast_models import ASTModel

def print_step(msg):
    print(f"\n{'='*10} {msg} {'='*10}\n")

def find_audio_file(audio_root, file_name):
    file_name = os.path.basename(file_name.strip().strip(' []{}"\'\n\r'))
    for root, dirs, files in os.walk(audio_root):
        if file_name in files:
            return os.path.join(root, file_name)
    base, _ = os.path.splitext(file_name)
    for ext in ['.wav', '.mp3', '.flac']:
        candidate = base + ext
        for root, dirs, files in os.walk(audio_root):
            if candidate in files:
                return os.path.join(root, candidate)
    raise FileNotFoundError(f"Audio file '{file_name}' not found under {audio_root}")

def robust_audio_load(path, min_length=64000):
    path = os.path.abspath(path)
    try:
        wav, sr = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = torch.tensor(wav, dtype=torch.float32)
        print(f"Loaded audio via soundfile: {path}")
    except Exception:
        try:
            wav, sr = torchaudio.load(path)
            wav = wav.mean(dim=0)
            print(f"Loaded audio via torchaudio: {path}")
        except Exception:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', path,
                    '-ar', '16000', '-ac', '1', '-f', 'wav', tmp_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                wav, sr = torchaudio.load(tmp_path)
                wav = wav.mean(dim=0)
                print(f"Converted audio via ffmpeg: {path}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    if wav.shape[0] < min_length:
        wav = torch.nn.functional.pad(wav, (0, min_length - wav.shape[0]))
    return wav, sr

class ASVspoofDataset(Dataset):
    def __init__(self, trials, audio_root, transform=None, min_audio_len=64000, tdim=100):
        self.trials = trials
        self.audio_root = audio_root
        self.transform = transform
        self.min_audio_len = min_audio_len
        self.tdim = tdim
        print(f"ASVspoofDataset initialized with {len(trials)} trials, audio_root={audio_root}")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        file_entry, label = self.trials[idx]
        if os.path.isabs(file_entry):
            audio_path = file_entry
        else:
            audio_path = find_audio_file(self.audio_root, file_entry)
        print(f"Loading item {idx}: {audio_path}, label={label}")
        wav, sr = robust_audio_load(audio_path, min_length=self.min_audio_len)
        spec = self.transform(wav) if self.transform else wav.unsqueeze(0)
        if spec.shape[1] < self.tdim:
            spec = torch.nn.functional.pad(spec, (0, self.tdim - spec.shape[1]))
        elif spec.shape[1] > self.tdim:
            spec = spec[:, :self.tdim]
        assert spec.shape[0] == 128 and spec.shape[1] == self.tdim, \
            f"Expected spec [128,{self.tdim}], got {spec.shape}"
        return spec, label

def collate_fn(batch):
    print(f"Collating batch of size {len(batch)}")
    specs, labels = zip(*batch)
    x = torch.stack(specs).transpose(1, 2)
    y = torch.tensor(labels)
    return x, y

def load_trials(meta_path):
    trials = []
    print(f"Loading trials from {meta_path}")
    with open(meta_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            fname = parts[1]
            label = 0 if parts[-1].lower() == 'bonafide' else 1
            trials.append((fname, label))
    print(f"Loaded {len(trials)} ASVspoof trials")
    return trials

def load_audio_mnist_trials(meta_path):
    print(f"Reading AudioMNIST metadata: {meta_path}")
    trials = []
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    meta_dir = os.path.dirname(os.path.abspath(meta_path))
    for speaker_id in meta:
        speaker_subdir = speaker_id.strip().zfill(2)
        speaker_path = os.path.join(meta_dir, speaker_subdir)
        if os.path.isdir(speaker_path):
            files = [f for f in os.listdir(speaker_path) if f.lower().endswith('.wav')]
            print(f"Found {len(files)} .wav files in {speaker_subdir}")
            for file in files:
                trials.append((os.path.join(speaker_path, file), 0))
    print(f"Total AudioMNIST trials: {len(trials)}")
    return trials

def load_ai_tts_trials(tts_root):
    print(f"Walking AI-TTS directory: {tts_root}")
    trials = []
    abs_root = os.path.abspath(tts_root)
    for root, dirs, files in os.walk(abs_root):
        for fname in files:
            if fname.lower().endswith(('.wav', '.mp3')):
                trials.append((os.path.join(root, fname), 1))
    print(f"Total AI-TTS trials: {len(trials)}")
    return trials

def find_file(root, filename):
    for r, dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(r, filename)
    return None

def find_dir_by_keyword(root, keyword):
    for r, dirs, files in os.walk(root):
        for d in dirs:
            if keyword.lower() in d.lower():
                return os.path.join(r, d)
    return None

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-meta', required=True)
    p.add_argument('--dev-meta', required=True)
    p.add_argument('--audio-root', required=True)
    p.add_argument('--pretrained-path', required=True)
    p.add_argument('--model-size', choices=['tiny','base','large'], default='tiny')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--subset-fraction', type=float, default=1.0)
    p.add_argument('--output-dir', type=str, default='./finetuned_models')
    p.add_argument('--test-root', type=str, default=None,
                   help='Directory containing audioMNIST_meta.txt and tts subfolder')
    return p.parse_args()

def main():
    args = get_args()
    print(f"Parsed arguments: {args}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_step(f"Using device: {device}")

    print_step("Loading finetuning trials")
    train_trials = load_trials(args.train_meta)
    dev_trials = load_trials(args.dev_meta)

    if args.subset_fraction < 1.0:
        n_tr = int(len(train_trials) * args.subset_fraction)
        n_dev = int(len(dev_trials) * args.subset_fraction)
        train_trials = train_trials[:n_tr]
        dev_trials = dev_trials[:n_dev]
        print_step(f"Subset applied: {len(train_trials)} train, {len(dev_trials)} dev")

    tdim = 100
    transform = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(16000, 400, 160, 128),
        torchaudio.transforms.AmplitudeToDB()
    )

    print_step("Creating DataLoaders")
    train_loader = DataLoader(
        ASVspoofDataset(train_trials, args.audio_root, transform, 64000, tdim),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(
        ASVspoofDataset(dev_trials, args.audio_root, transform, 64000, tdim),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print_step("Initializing SSAST model")
    model = ASTModel(
        label_dim=2,
        fstride=128,
        tstride=2,
        input_fdim=128,
        input_tdim=tdim,
        model_size=args.model_size,
        pretrain_stage=False,
        load_pretrained_mdl_path=args.pretrained_path,
    ).to(device)
    print(f"Model initialized: size={args.model_size}, pretrain_stage=False")
    print_step(f"Loaded checkpoint from {args.pretrained_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    print_step("Starting training loop")
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs} - Training phase")
        model.train()
        total_loss, count = 0.0, 0
        for i, (specs, labels) in enumerate(tqdm(train_loader, desc="Train", leave=False)):
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(specs, task="ft_avgtok")
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * specs.size(0)
            count += specs.size(0)
            print(f"  [Train] Batch {i+1}/{len(train_loader)}, loss={loss.item():.4f}")
        avg_train_loss = total_loss / count
        print_step(f"Epoch {epoch} Training complete, Avg Loss: {avg_train_loss:.4f}")

    print_step("Saving fine-tuned model")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"finetuned_{args.model_size}.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")

    if args.test_root:
        print_step("Starting test evaluation")
        meta_path = find_file(args.test_root, 'audioMNIST_meta.txt')
        if not meta_path:
            print_step("audioMNIST_meta.txt not found; skipping test")
            return
        audio_trials = load_audio_mnist_trials(meta_path)
        ai_trials = load_ai_tts_trials(find_dir_by_keyword(args.test_root, 'tts') or '')
        test_trials = audio_trials + ai_trials
        print(f"Total test trials: {len(test_trials)} (AudioMNIST={len(audio_trials)}, AI-TTS={len(ai_trials)})")

        print_step("Creating test DataLoader")
        test_loader = DataLoader(
            ASVspoofDataset(test_trials, args.test_root, transform, 64000, tdim),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        print(f"Beginning test inference")
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for i, (specs, labels) in enumerate(tqdm(test_loader, desc="Test", leave=False)):
                specs, labels = specs.to(device), labels.to(device)
                preds = model(specs, task="ft_avgtok").argmax(1)
                correct += (preds == labels).sum().item()
                total += specs.size(0)
                print(f"  [Test] Batch {i+1}/{len(test_loader)}, Acc so far: {correct/total:.4f}")
        print_step(f"Test complete, Final Accuracy: {correct/total:.4f} ({correct}/{total})")
    else:
        print_step("Skipping test (no --test-root provided)")

if __name__ == '__main__':
    main()
