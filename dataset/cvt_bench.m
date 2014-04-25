%convert benchmarks.mat into separate mat files
data = load('benchmarks');
names = fieldnames(data);
% h5create('bench.h5'
for i = 1:numel(names)-1
    dt = data.(names{i});
    x = dt.x;
    t = dt.t;
    train = dt.train;
    test = dt.test;
%     h5create('bench.h5', ['/',names{i}] 
    save([names{i},'mat'],'x','t','train','test');
end
