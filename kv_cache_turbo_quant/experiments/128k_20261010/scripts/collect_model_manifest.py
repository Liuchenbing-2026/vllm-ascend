import json,hashlib,pathlib
def main():
    root=pathlib.Path('/ws')
    model=pathlib.Path('/models/Qwen3-30B-A3B')
    names=['config.json','tokenizer_config.json','model.safetensors.index.json','tokenizer.json','generation_config.json']
    files={name:{'sha256':hashlib.sha256((model/name).read_bytes()).hexdigest(),'bytes':(model/name).stat().st_size} for name in names if (model/name).exists()}
    out={'path':str(model),'files':files,'weight_files':[{'file':p.name,'bytes':p.stat().st_size} for p in sorted(model.glob('*.safetensors'))],'weight_hashes':'not computed; local preexisting model, origin revision unverified'}
    (root/'artifacts/model-manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
