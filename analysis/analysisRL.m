function analysisRL()

cd('data/learning_data') % Go to directory with raw behavioral data files

analyse4AFCProbeData("w", "Probe_4AFC", true)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analyse4AFCProbeData(subject, task, saveFiles)

options = optimset('MaxIter', 10000, 'MaxFunEvals', 10000, 'TolFun', 0.00001);

%%% Probe Data

datPath = subject + "_" + task + ".csv";

dat = readtable(datPath);

nStim = sum(contains(dat.Properties.VariableNames, 'stimchoice'));

nCues = length(unique(dat.stimchoice1));

ncol = size(dat, 2);

%%% Split trials into choose matching color and choose matching shape

colorChoicei = dat.is_choice_color == 1;
shapeChoicei = dat.is_choice_color == 0;
colorChoice = dat(colorChoicei, :);
shapeChoice = dat(shapeChoicei, :);

%%% Initialize table for saving RL parameters
paramsTable = table('Size',[2 19],'VariableTypes',["string","double","double", ...
    "double","double","double","double","double","double","double","double","double", ...
    "double","double","double","double","double","double","double"], ...
    'VariableNames',["trial_type", ...
    "beta", "beta_SE", "beta_CI", "beta_ucb", "beta_lcb", ...
    "alpha", "alpha_SE", "alpha_CI", "alpha_ucb", "alpha_lcb", ...
    "v0", "v0_SE", "v0_CI", "v0_ucb", "v0_lcb", ...
    "p0", "p0_ucb", "p0_lcb"]); 

%%% run RL model on each condition
for trialTypei = 1 : 2

    if trialTypei == 1
        tdat = colorChoice;
        trialName = "choose_color";
    else
        tdat = shapeChoice;
        trialName = "choose_shape";
    end
    paramsTable(trialTypei,1) = {trialTypei};

    state  = tdat.correct_stim;
    choice = tdat.stimchosen; 
    rew    = tdat.chose_correct; 

    %%% initial values for parameters
    params = [3 0.1 0.8];

    [mparams, lla] = fminsearch(@(params) rlFit(params, state, choice, rew, nCues), ...
        params, options); 

    [~, pChoice] = rlFitf(mparams, state, choice, rew, nCues); 
    
    %%% compute hessian to get standard deviation of parameter estimates.
    %%% because hessian is not always positive definite, use smallest
    %%% derivative that works
    
    for expn = -10 : 1 : -4 % freeze model, vary optimized parameters to get estimate of variance

        epsi = 10^(expn);
    
        h = evaluateHessian(mparams, state, choice, rew, nCues, epsi);

        if min(diag(h)) > 0 % stop searching once you have a matrix with an all positive diagonal
            break
        end
    end

    ih = inv(h);

    varsd = sqrt(diag(ih)); 

    for vari = 1 : length(mparams)

        %%% upper and lower 95% bounds for parameters
        ucb = mparams(vari) + varsd(vari)*1.96; %%% 2.58 for 0.01, 1.96 for 0.05 two sided
        lcb = mparams(vari) - varsd(vari)*1.96;
        fprintf('Trialtypei %d var %d mean %.5f %.7f %.5f %.5f\n', trialTypei, vari, mparams(vari), varsd(vari), ucb, lcb);
        ci = varsd(vari)*1.96;
        paramsTable(trialTypei,vari*5-3:vari*5+1) = {mparams(vari), varsd(vari), ci, ucb, lcb};

        %%% check if initial probability is above chance
        if vari == 3

            pvec = [lcb zeros(1, nCues)];

            pc_l = exp(mparams(1)*pvec)/sum(exp(mparams(1)*pvec));

            fprintf('initial prob %.4f\n', pc_l(1));

            pc_u = exp(mparams(1)*[ucb zeros(1, nCues)])/sum(exp(mparams(1)*[ucb zeros(1, nCues)]));
            pc =  exp(mparams(1)*[mparams(vari) zeros(1, nCues)])/sum(exp(mparams(1)*[mparams(vari) zeros(1, nCues)]));
            paramsTable(trialTypei,17:19) = {pc(1), pc_u(1), pc_l(1)};

        end

    end

    %%% compute moving averages for plots
    avgBins = 100;

    mvAvgChoice = zeros(size(tdat, 1) - avgBins+1, 1);
    mvAvgModel  = zeros(size(tdat, 1) - avgBins+1, 1);
    nTrials = size(mvAvgChoice, 1);

    for triali = avgBins : size(tdat, 1)

        mvAvgChoice(triali-avgBins + 1) = mean(tdat.chose_correct((triali - avgBins + 1) : triali));
        mvAvgModel(triali-avgBins + 1)  = mean(pChoice((triali-avgBins+1) : triali));

    end  

    %%% save data
    if saveFiles == true
        save(subject+"_"+task+"_"+trialName+"_RLfit.mat", "mvAvgChoice", "mvAvgModel", "pChoice");
    end

    %%% plot moving average of choice and model
    subplot(2,2,trialTypei+2)
    plot(1:nTrials, mvAvgChoice, 1:size(mvAvgModel, 1), mvAvgModel);
    title([mparams lla])

%%% save table with RL parameters
if saveFiles == true
    writetable(paramsTable, subject + "_" + task + "_RL_params.csv");
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ll = rlFit(params, state, choice, rew, nCues) % returns negative log likelihood 

[ll, ~] = rlFitf(params, state, choice, rew, nCues);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ll, tpChoice] = rlFitf(params, state, choice, rew, nCues)

nTrials = length(state);

beta = params(1); % inverse temperature 
lr1  = params(2); % learning rate 
lr2  = lr1;
v0   = params(3); % initial expected reward

lr1a = 1;

v = 0.0*ones(nCues, nCues) + v0*diag(ones(nCues, 1));

pchoice = zeros(nTrials, nCues);
tpChoice = zeros(nTrials, 1);

ll = 0; 

for triali = 1 : nTrials

    pchoice(triali, :) = exp(beta*v(state(triali), :))/sum(exp(beta*v(state(triali), :)));

    tpChoice(triali) = pchoice(triali, state(triali)); %% probability of correct choice, not actual choice

    ll = ll - log(pchoice(triali, choice(triali))); 

    lr1 = lr1a*lr1;
    
    if rew(triali) == 1 % update expected reward for a given state
        v(state(triali), choice(triali)) = v(state(triali), choice(triali)) + lr1*(rew(triali) - v(state(triali), choice(triali)));
    else
        v(state(triali), choice(triali)) = v(state(triali), choice(triali)) + lr2*(rew(triali) - v(state(triali), choice(triali)));
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = evaluateHessian(params, state, choice, rew, nCues, epsi)

N = length(params);

for v1 = 1 : N
    for v2 = v1 : N
        
        ev = zeros(1,N);
        ev(v1) = epsi;
        ev(v2) = ev(v2) + epsi;
        ll1 = rlFit(params+ev, state, choice, rew, nCues);
        
        ev = zeros(1,N);
        ev(v1) = epsi;
        ev(v2) = ev(v2) - epsi;
        ll2 = rlFit(params+ev, state, choice, rew, nCues);
        
        ev = zeros(1,N);
        ev(v1) = -epsi;
        ev(v2) = ev(v2) + epsi;
        ll3 = rlFit(params+ev, state, choice, rew, nCues);
        
        ev = zeros(1,N);
        ev(v1) = -epsi;
        ev(v2) = ev(v2) - epsi;
        ll4 = rlFit(params+ev, state, choice, rew, nCues);
        
        h(v1, v2) = (1/(4*epsi^2))*(ll1 - ll2 - ll3 + ll4); 
        h(v2, v1) = h(v1, v2);
        
    end
end
