###########################################

incident_wave_regions = [1 2 3]
scatterer_regions = [2 3]

###########################################

function geometry(x,y,∂,inputs::Dict)
    
    extras = inputs["extras"]

    origin = extras[1]
    
    R = extras[3]
    β = extras[4]

    d1 = R*0.01
    d2 = R*0.01
    d3 = R*0.01

    R1 = 0.043*R
    R2 = 0.048285714285714286*R
    R3 = 0.048285714285714286*R

    origin1 = [1 , 0]*(R+d1+R1)
    origin2 = [cos(β) , sin(β)]*(R+d2+R2)
    origin3 = [cos(sqrt(2)*β) , sin(sqrt(2)*β)]*(R+d3+R3)
    
    local region::Int
        
    if ( (x-origin[1])^2 + (y-origin[2])^2 < R^2)
        region = 2
#    elseif R+d1 < (x-origin[1]) < R+d1+2R1
    elseif ( (x-origin1[1])^2 + (y-origin1[2])^2 < R1^2)
        region = 3
    elseif ( (x-origin2[1])^2 + (y-origin2[2])^2 < R2^2)
        region = 3
    elseif ( (x-origin3[1])^2 + (y-origin3[2])^2 < R3^2)
        region = 3
    else
        region = 1
    end
    
    return region

end

return incident_wave_regions, scatterer_regions, geometry