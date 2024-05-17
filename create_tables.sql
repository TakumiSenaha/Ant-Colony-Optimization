-- Parameters table
CREATE TABLE parameters (
    parameterid SERIAL PRIMARY KEY,
    numberofnodes INTEGER,
    optimalpathlength INTEGER,
    volatility FLOAT,
    minpheromone INTEGER,
    maxpheromone INTEGER,
    ttl INTEGER,
    bata INTEGER,
    generationlimit INTEGER,
    UNIQUE (numberofnodes, optimalpathlength, volatility, minpheromone, maxpheromone, ttl, bata, generationlimit)
);

-- Simulations table
CREATE TABLE simulations (
    simulationid SERIAL PRIMARY KEY,
    parameterid INTEGER REFERENCES parameters(parameterid)
);

-- Nodes table
CREATE TABLE nodes (
    nodeid SERIAL PRIMARY KEY,
    simulationid INTEGER REFERENCES simulations(simulationid),
    num_of_connections INTEGER
);

-- Generations table
CREATE TABLE generations (
    generationid SERIAL PRIMARY KEY,
    simulationid INTEGER REFERENCES simulations(simulationid),
    generation_count INTEGER
);

-- Connections table
CREATE TABLE connections (
    connectionid SERIAL PRIMARY KEY,
    generationid INTEGER REFERENCES generations(generationid),
    startnodeid INTEGER REFERENCES nodes(nodeid),
    endnodeid INTEGER REFERENCES nodes(nodeid),
    pheromone FLOAT,
    width INTEGER
);

-- Ants table
CREATE TABLE ants (
    antid SERIAL PRIMARY KEY,
    generationid INTEGER REFERENCES generations(generationid),
    sourcenodeid INTEGER REFERENCES nodes(nodeid),
    destinationnodeid INTEGER REFERENCES nodes(nodeid),
    routenodesid INTEGER[],
    routewidths INTEGER[],
    routebottleneck INTEGER
);

-- Interests table
CREATE TABLE interests (
    interestid SERIAL PRIMARY KEY,
    generationid INTEGER REFERENCES generations(generationid),
    sourcenodeid INTEGER REFERENCES nodes(nodeid),
    destinationnodeid INTEGER REFERENCES nodes(nodeid),
    routenodesid INTEGER[],
    routewidths INTEGER[],
    routebottleneck INTEGER
);
