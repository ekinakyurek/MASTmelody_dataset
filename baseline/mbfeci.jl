function acc(w, data, state)
    sumloss = numloss = 0

    for d in data
        x = d[1];
        y = d[2];

        per = x[:,1:1000];
        ref = x[:,1001:2000];

        z = model(w, per, ref, copy(state))

        sumloss += mean((z .* y) .> 0)
        numloss += 1
    end
    sumloss/numloss
end

function init_rnn_weights(hidden, embed)
    model = Array(Any, 2*length(hidden)+4)
    X = embed
    for k = 1:length(hidden)
        H = hidden[k]
        model[2k-1] = xavier(X+H, 4H)
        model[2k] = zeros(1, 4H)
        model[2k][1:H] = 1 # forget gate bias = 1
        X = H
    end

    model[end-1] = 0.1*randn(2*hidden[end],1)
    model[end] = zeros(1,1)

    model[end-3] = 0.1*randn(100,100)
    model[end-2] = zeros(1,100)
    #model[end] = xavier(vocab,embed)    #We (word embedding vector)
    return model
end

# state[2k-1]: hidden for the k'th lstm layer
# state[2k]: cell for the k'th lstm layer
function initstate(hidden_layers, batchsize,atype=Array{Float32})
    nlayers = length(hidden_layers);
    state = Array(Any, 2*nlayers);
    for k = 1:nlayers
        state[2k-1] = zeros(Float32, batchsize, hidden_layers[k]);
        state[2k] = zeros(Float32, batchsize, hidden_layers[k]);
    end
    return map(k->convert(atype,k), state)
end

function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function rnn(w,s,input; start = 0)
    for i=1:2:length(s)
        #input = dropout(input,pdrop)
        (s[i],s[i+1]) = lstm(w[start + i],w[start + i+1],s[i],s[i+1],input)
        input = s[i]
    end
    #input = dropout(input,pdrop)
    return input
end

function loss(w,x,y,z,s)
    ypred = model(w,x,y,s)
    mean(log(1 .+ exp(-z .* ypred)))
end

function model(w,x,y,s)
    xfeat = 0
    s1 = copy(s)

    for i in range(1,10)
        xfeat = rnn(w,s1,x[:,100*(i-1)+1:100*i]*w[end-3].+w[end-2])
    end
    s2 = copy(s)
    yfeat = 0
    for i in range(1,10)
        yfeat = rnn(w,s2,y[:,100*(i-1)+1:100*i]*w[end-3].+w[end-2])
    end
    hcat(xfeat,yfeat)*w[end-1] .+ w[end]
end

lgrad = grad(loss);

function train!(m, data, state, opts)
    count = 0
    for d in data
        x = d[1];
        y = d[2];

        per = x[:,1:1000];
        ref = x[:, 1001:2000];

        dw = lgrad(m, per, ref, y, copy(state))

        for i in 1:length(m)
            update!(m[i], dw[i], opts[i])
        end

        count += 1
        #println(count)
    end
end

function init_params(model)
    prms = Array(Any, length(model))
    for i in 1:length(model)
        prms[i] = Adam()
    end
    return prms
end

function modelrun(data; epochs=100, hidden = [256])
    w = init_rnn_weights(hidden, 100)
    state = initstate(hidden, 100)
    opts = init_params(w);

    println("Initialized model")
    println("Accuracies for: train-dev sets:")

    msg(e) = println((e,map(d->acc(w,d,state),data)...));
    msg(0)

    for epoch = 1:epochs
      # Alternative: one could keep the model with highest accuracy in development set results
      # and return that one instead of the last model
        train!(w, data[1], state, opts)#training on the train set (data[1])
        msg(epoch)
    end
    return model, state
end
