source /usr/local/Ascend/ascend-toolkit/set_env.sh
$ASCEND_HOME_PATH/python/site-packages/bin/msopgen gen -i bert_self_attention.json -f pytorch -c ai_core-Ascend910B4 -lan cpp -out BertSelfAttention_
cp -r ./BertSelfAttention/* ./BertSelfAttention_
rm -rf ./BertSelfAttention/*
mv ./BertSelfAttention_/* ./BertSelfAttention
rm -rf BertSelfAttention_
cd ./BertSelfAttention
if [ ! -f "CMakePresets.json" ]; then
    echo "Error: CMakePresets.json File does not exist."
    exit 1
fi
patch -p1 CMakePresets.json < CMakePresets.patch
patch -p1 cmake/util/makeself/makeself-header.sh < cmake.patch
sed -i 's|"customize"|"mxRAG"|' "CMakePresets.json"
sed -i 's|TMPROOT=\\${TMPDIR:="\\$HOME"}|TMPROOT="\\$PWD"|' cmake/util/makeself/makeself-header.sh
chmod -R +x ./build.sh
find . -type f -name "*.sh" -print0 | xargs -0 dos2unix
bash build.sh
./build_out/custom_opp_*.run
cd ..