root = string(fileparts(mfilename("fullpath")));
figdir = fullfile(root, "figures");
if ~exist(figdir, "dir"); mkdir(figdir); end

S = load(fullfile(root, "trial1", "mnist.mat"));
fprintf("training fields: %s\n", strjoin(fieldnames(S.training), ","));
fprintf("images class=%s size=%s min=%g max=%g\n", class(S.training.images), mat2str(size(S.training.images)), min(S.training.images(:)), max(S.training.images(:)));
fprintf("train count=%d test count=%d\n", S.training.count, S.test.count);

Xtr = double(reshape(S.training.images, 784, [])); ytr = double(S.training.labels(:));
Xte = double(reshape(S.test.images, 784, []));     yte = double(S.test.labels(:));

trials = struct( ...
  "name", {"trial1","trial2","trial3","trial4","trial5","trial6"}, ...
  "n",    {[784 64 32 10],[784 256 128 64 10],[784 64 32 10],[784 64 32 10],[784 64 32 10],[784 64 32 10]}, ...
  "act",  {"relu","relu","sigmoid","relu","relu","relu"}, ...
  "desc", {"Baseline (ReLU+CE+Adam, bs64)","Deeper/wider [256 128 64]","Sigmoid hidden","MSE loss","SGD (no Adam)","Batch size 16"});

relu = @(x) max(0,x);
sigm = @(x) 1./(1+exp(-x));
fwd = @(W,b,X,L,act) forward(W,b,X,L,act,relu,sigm);

curves = cell(6,1); accTe = cell(6,1); lrs = cell(6,1);
fid = fopen(fullfile(figdir, "results.txt"), "w");
for k = 1:6
  t = trials(k); L = numel(t.n);
  d = fullfile(root, t.name, "WBtrain");
  files = dir(fullfile(d, "*.mat"));
  nE = numel(files);
  at = zeros(nE,1); lr = zeros(nE,1);
  for e = 1:nE
    C = load(fullfile(d, sprintf("WB%04d.mat", e)), "W", "b", "lr");
    W = cellfun(@double, C.W, "UniformOutput", false); b = cellfun(@double, C.b, "UniformOutput", false);
    [~, p] = max(fwd(W,b,Xte,L,t.act), [], 1);
    at(e) = mean(p(:)-1 == yte);
    lr(e) = C.lr;
  end
  C = load(fullfile(d, sprintf("WB%04d.mat", nE)));
  W = cellfun(@double, C.W, "UniformOutput", false); b = cellfun(@double, C.b, "UniformOutput", false);
  [~, ptr] = max(fwd(W,b,Xtr,L,t.act), [], 1);
  aTr = mean(ptr(:)-1 == ytr);
  [~, pte] = max(fwd(W,b,Xte,L,t.act), [], 1); pte = pte(:)-1;
  nParams = sum(cellfun(@numel, C.W)) + sum(cellfun(@numel, C.b));
  [bestAcc, bestE] = max(at);
  line = sprintf("%s | %s | params=%d | epochs=%d | final cost=%.4e | train acc=%.4f | test acc(final)=%.4f | best test acc=%.4f @ epoch %d | final lr=%.2e | lr reductions=%d", ...
    t.name, t.desc, nParams, nE, C.cost_list(end), aTr, at(end), bestAcc, bestE, lr(end), sum(diff(lr)<0));
  fprintf("%s\n", line); fprintf(fid, "%s\n", line);
  curves{k} = double(C.cost_list(:)); accTe{k} = at; lrs{k} = lr;

  if k == 1
    cm = confusionmat(yte, pte);
    f = figure("Visible","off","Position",[100 100 640 560]); theme(f,"light");
    confusionchart(cm, string(0:9), "Title", "Trial 1 Confusion Matrix (test set)", "RowSummary","row-normalized");
    exportgraphics(f, fullfile(figdir, "confusion_trial1.png"), "Resolution", 150, "BackgroundColor", "white"); close(f);
    fprintf(fid, "trial1 confusion matrix:\n%s\n", mat2str(cm));
    wrong = find(pte ~= yte);
    f = figure("Visible","off","Position",[100 100 900 380]); theme(f,"light");
    tiledlayout(2,8,"TileSpacing","compact","Padding","compact");
    for i = 1:16
      nexttile; idx = wrong(i);
      imshow(reshape(Xte(:,idx),28,28), []);
      title(sprintf("true %d / pred %d", yte(idx), pte(idx)), "FontSize", 8);
    end
    exportgraphics(f, fullfile(figdir, "misclassified_trial1.png"), "Resolution", 150, "BackgroundColor", "white"); close(f);
    f = figure("Visible","off","Position",[100 100 900 380]); theme(f,"light");
    tiledlayout(2,8,"TileSpacing","compact","Padding","compact");
    for i = 1:16
      nexttile; imshow(reshape(Xtr(:,i),28,28), []); title(sprintf("label %d", ytr(i)), "FontSize", 8);
    end
    exportgraphics(f, fullfile(figdir, "mnist_samples.png"), "Resolution", 150, "BackgroundColor", "white"); close(f);
  end
end
fclose(fid);

names = arrayfun(@(t) sprintf("%s: %s", t.name, t.desc), trials, "UniformOutput", false);

f = figure("Visible","off","Position",[100 100 900 520]); theme(f,"light");
hold on; for k=1:6; semilogy(curves{k}, "LineWidth", 1.4); end; hold off
set(gca, "YScale", "log"); grid on; xlabel("Epoch"); ylabel("Training cost"); title("Training cost per epoch");
legend(names, "Location", "northeast", "Interpreter", "none");
exportgraphics(f, fullfile(figdir, "cost_curves.png"), "Resolution", 150, "BackgroundColor", "white"); close(f);

f = figure("Visible","off","Position",[100 100 900 520]); theme(f,"light");
hold on; for k=1:6; plot(accTe{k}*100, "LineWidth", 1.4); end; hold off
grid on; xlabel("Epoch"); ylabel("Test accuracy (%)"); title("Test accuracy per epoch checkpoint"); ylim([floor(min(cellfun(@min, accTe))*100) 100]);
legend(names, "Location", "southeast", "Interpreter", "none");
exportgraphics(f, fullfile(figdir, "test_accuracy.png"), "Resolution", 150, "BackgroundColor", "white"); close(f);

f = figure("Visible","off","Position",[100 100 900 420]); theme(f,"light");
hold on; for k=1:6; semilogy(lrs{k}, "LineWidth", 1.4); end; hold off
set(gca, "YScale", "log"); grid on; xlabel("Epoch"); ylabel("Learning rate"); title("Learning rate schedule (ReduceLROnPlateau)"); ylim([4e-5 1.5e-3]);
legend(names, "Location", "southwest", "Interpreter", "none");
exportgraphics(f, fullfile(figdir, "lr_schedule.png"), "Resolution", 150, "BackgroundColor", "white"); close(f);

function a = forward(W,b,X,L,act,relu,sigm)
  a = X;
  for l = 2:L-1
    z = W{l}*a + b{l};
    if act == "relu"; a = relu(z); else; a = sigm(z); end
  end
  z = W{L}*a + b{L}; z = z - max(z,[],1);
  a = exp(z)./sum(exp(z),1);
end
