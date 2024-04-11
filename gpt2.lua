require("math")


function encoder_forward(out, inp, wte, wpe, B, T, C)
  for b = 0, B - 1 do
    for t = 0, T - 1 do
      -- seek to the output position
      local out_bt = out + b * T * C + t * C

      -- get the index of the token at inp[b, t]
      local input_index = (b - 1) * T + t
      local ix = inp[input_index]

      -- seek to the position in wte corresponding to the token
      local wte_ix = wte + ix * C
      -- seel to the position in wpe corresponding to the position
      local wpe_t = wpe + t * C

      -- add the two vectors
     for c=0, C - 1 do
        out_bt[c] = wte_ix[c] + wpe_t[c]
      end

    end
  end
end

function encoder_backward(dwte, dwpe, dout, inp, B, T, C)
  for b=0, B - 1 do
    for t=0, T - 1 do
      -- seek to the output position
      local dout_bt = dout + b * T * C + t * C

      -- get the index of the token at inp[b, t]
      local input_index = (b - 1) * T + t
      local ix = inp[input_index]

      -- seek to the position in dwte corresponding to the token
      local dwte_ix = dwte + ix * C
      -- seel to the position in dwpe corresponding to the position
      local dwpe_t = dwpe + t * C

      -- add the two vectors
      for c=0, C - 1 do
        local d = dout_bt[c]
        dwte_ix[c] = dwte_ix[c] + d
        dwpe_t[c] = dwpe_t[c] + d
      end
    end
  end
end

function layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C)
  local eps = 1e-5
  for b=0, B - 1 do
    for t=0, T - 1 do
      local x = inp + b * T * C + t * C
      local m = 0.0

      for c=0, C - 1 do
        m = m + x[c]
      end
      m= m / C

      local v = 0.0

      for c=0, C - 1 do
        local xshift = x[c] - m
        v = v + xshift * xshift
      end
      v = v / C

      -- calculate the rstd
      local s = 1.0 / math.sqrt(v + eps)

      local outbt = out + b * T * C + t * C
      for c=0, C - 1 do
        local n = (s * (x[c] - m))
        local o = (n * weight[c] + bias[c])
        outbt[c] = o
      end

      mean[b * T + t] = m
      rstd[b * T + t] = s
    end
  end
end

-- TODO implement layernorm_backward
function layernorm_backward(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C)
 for b=0, B - 1 do
  for t=0, T -1 do
    local dout_bt = dout + b * T * C + t * C
    local inp_bt = inp + b * T * C + t * C
    local dinp_bt = dinp + b * T * C + t * C
    local mean_bt = mean[b * T + t]
    local rstd_bt = rstd[b * T + t]

    -- first: two reduce operations
    local dnorm_mean = 0.0
    local dnorm_norm_mean = 0.0
    for c=0, C - 1 do
       local norm_bti = (inp_bt[c] - mean_bt) * rstd_bt
        local dnorm_i = weight[c] * dout_bt[c]
    end

  end
 end
end
