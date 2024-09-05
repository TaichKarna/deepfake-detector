const signUp = async (userData) => {

    const res = await fetch(`${import.meta.env.VITE_API}/auth/signup`,{
        method: 'POST',
        headers:{
            "Content-Type": "application/json"
        },
        body: JSON.stringify(userData)
    });

    const data = await res.json();

    if(!res.ok){
        throw new Error(data.detail);
    }

    return data;
}

const signIn = async (userData) => {

    const res = await fetch(`${import.meta.env.VITE_API}/auth/login`,{
        method: 'POST',
        headers:{
            "Content-Type": "application/json"
        },
        body: JSON.stringify(userData)
    });

    const data = await res.json();
    
    if(!res.ok){
        throw new Error(data.detail);
    }

    return data;
}

export { signUp, signIn };